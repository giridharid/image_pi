"""
GS1 DataKart Intelligence API — Acquink Solutions
Follows exact same pattern as Smaartbrand Hotels agent v2 (main.py)
Auth: GCP_CREDENTIALS_JSON env var
Agent: geminidataanalytics_v1alpha (agent_fed5f0dd-53ec-4529-b3c1-c01f626201dc)
"""

from fastapi import FastAPI, HTTPException
from fastapi.responses import HTMLResponse, FileResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Optional, List
import json, os, base64, traceback, threading, time, uuid, math

from agent_prompts import get_agent_prompt, DATA_CONTEXT_TEMPLATE, DATASET_CONTEXT_TEMPLATE


def clean_val(v):
    if v is None: return None
    if hasattr(v, 'isoformat'): return v.isoformat()
    if hasattr(v, 'item'): v = v.item()
    if isinstance(v, float) and (math.isnan(v) or math.isinf(v)): return None
    return v

def clean_row(d):
    result = {}
    for k, v in d.items():
        cv = clean_val(v)
        # Strip .0 float suffix from ID fields
        if k in ('barcode','fssai','acquink_id') and cv is not None:
            s = str(cv).strip()
            if s.endswith('.0') and s[:-2].isdigit():
                cv = s[:-2]
        result[k] = cv
    return result


app = FastAPI(title="GS1 DataKart Intelligence API", version="1.0")
app.add_middleware(CORSMiddleware, allow_origins=["*"],
                   allow_credentials=True, allow_methods=["*"], allow_headers=["*"])

PROJECT        = "gen-lang-client-0143536012"
DATASET        = "gs1_datakart"
TABLE          = "products"
AGENT_ID       = os.environ.get("GEMINI_AGENT_ID",
                 "agent_fed5f0dd-53ec-4529-b3c1-c01f626201dc")
AGENT_LOCATION = "global"
GCS_BUCKET     = os.environ.get("GCS_BUCKET", "gs1-datakart-images")
GCS_BASE       = f"https://storage.googleapis.com/{GCS_BUCKET}"

CACHE = {
    "products": {"data": None, "timestamp": 0},
    "stats":    {"data": None, "timestamp": 0},
    "regions":  {"data": None, "timestamp": 0},
}
CACHE_TTL = 86400

def cache_valid(k):
    return CACHE.get(k, {}).get("data") is not None and \
           (time.time() - CACHE[k]["timestamp"]) < CACHE_TTL
def get_cache(k): return CACHE[k]["data"] if cache_valid(k) else None
def set_cache(k, d): CACHE[k] = {"data": d, "timestamp": time.time()}


# ── Credentials (same as Hotels) ──────────────────────────────────────────────
bq_client    = None
gemini_model = None

def get_gcp_credentials():
    raw = os.environ.get("GCP_CREDENTIALS_JSON", "").strip().strip('"').strip("'")
    if not raw: return None, None
    try:
        from google.oauth2 import service_account
        if raw.startswith("{"):
            d = json.loads(raw)
        else:
            padding = 4 - len(raw) % 4
            if padding != 4: raw += "=" * padding
            d = json.loads(base64.b64decode(raw).decode("utf-8"))
        creds = service_account.Credentials.from_service_account_info(d)
        return creds, d.get("project_id", PROJECT)
    except Exception as e:
        print(f"[CREDS ERROR] {e}")
        return None, None

def init_bq_client():
    global bq_client
    if bq_client: return bq_client
    try:
        from google.cloud import bigquery
        creds, proj = get_gcp_credentials()
        if creds:
            bq_client = bigquery.Client(credentials=creds, project=proj)
        else:
            bq_client = bigquery.Client(project=PROJECT)
        print(f"[BQ] Connected: {PROJECT}.{DATASET}")
        return bq_client
    except Exception as e:
        print(f"[BQ ERROR] {e}"); return None

def init_gemini():
    global gemini_model
    creds, proj = get_gcp_credentials()
    if creds:
        try:
            import vertexai
            from vertexai.generative_models import GenerativeModel
            vertexai.init(project=proj, location="asia-south1", credentials=creds)
            for name in ["gemini-2.5-flash", "gemini-2.0-flash", "gemini-1.5-flash"]:
                try:
                    gemini_model = GenerativeModel(name)
                    print(f"[GEMINI FALLBACK] Vertex: {name}")
                    return gemini_model
                except: pass
        except Exception as e:
            print(f"[GEMINI] Vertex failed: {e}")
    api_key = os.environ.get("GEMINI_API_KEY", "")
    if api_key:
        import google.generativeai as genai
        genai.configure(api_key=api_key)
        gemini_model = genai.GenerativeModel("gemini-2.0-flash")
        print("[GEMINI FALLBACK] API key")
        return gemini_model
    print("[WARN] No Gemini fallback"); return None

def get_data_chat_client():
    try:
        from google.cloud import geminidataanalytics_v1alpha as gda
        creds, _ = get_gcp_credentials()
        return gda.DataChatServiceClient(credentials=creds) if creds \
               else gda.DataChatServiceClient()
    except Exception as e:
        print(f"[CHAT CLIENT] unavailable: {e}"); return None

def get_bq():
    global bq_client
    if not bq_client: init_bq_client()
    return bq_client

def get_gemini():
    global gemini_model
    if not gemini_model: init_gemini()
    return gemini_model


# ── Startup & caching (same pattern as Hotels) ─────────────────────────────────
@app.on_event("startup")
async def startup():
    init_bq_client()
    init_gemini()
    threading.Thread(target=load_caches, daemon=True).start()

NUTRIENT_COLS = [
    ("n_energy",            "n_energy_unit",            "Energy"),
    ("n_protein",           "n_protein_unit",           "Protein"),
    ("n_total_carbohydrate","n_total_carbohydrate_unit", "Total Carbohydrate"),
    ("n_total_sugars",      "n_total_sugars_unit",       "Total Sugars"),
    ("n_total_fat",         "n_total_fat_unit",          "Total Fat"),
    ("n_saturated_fat",     "n_saturated_fat_unit",      "Saturated Fat"),
    ("n_trans_fat",         "n_trans_fat_unit",          "Trans Fat"),
    ("n_sodium",            "n_sodium_unit",             "Sodium"),
    ("n_dietary_fiber",     "n_dietary_fiber_unit",      "Dietary Fiber"),
    ("n_calcium",           "n_calcium_unit",            "Calcium"),
    ("n_iron",              "n_iron_unit",               "Iron"),
]

GCS_BASE_URL = f"https://storage.googleapis.com/{GCS_BUCKET}"

def derive_images(row: dict) -> dict:
    """Derive image URLs from barcode or source file when columns are empty."""
    barcode = str(row.get("barcode") or "").strip()
    acquink = str(row.get("acquink_id") or "").strip()
    source  = str(row.get("file") or row.get("source_number") or "").strip()

    # For numbered images (ACQ-xxx), use source filename
    if acquink.startswith("ACQ-") and source:
        num = source.replace(".jpg","").replace(".png","").strip()
        img = f"{num}.jpg"
        if not row.get("product_image") or str(row.get("product_image","")) in ("nan","None",""):
            row["product_image"] = img
        return row

    # For barcoded products, derive from barcode
    if not barcode or barcode in ("nan","None",""):
        return row

    # Check image_files for the actual extension
    img_files = str(row.get("image_files") or "")
    ext = ".jpg" if ".jpg" in img_files else ".png"

    if not row.get("front_image") or str(row.get("front_image","")) in ("nan","None",""):
        row["front_image"] = f"{barcode}_f{ext}"
    if not row.get("back_image") or str(row.get("back_image","")) in ("nan","None",""):
        row["back_image"] = f"{barcode}_ba{ext}"
    if not row.get("product_image") or str(row.get("product_image","")) in ("nan","None",""):
        row["product_image"] = f"{barcode}_ba{ext}"

    # Fix _b.ext → _ba.ext
    for field in ("front_image","back_image","product_image"):
        v = str(row.get(field,"") or "")
        for e in (".png",".jpg"):
            if v.endswith(f"_b{e}"):
                row[field] = v[:-len(f"_b{e}")] + f"_ba{e}"

    return row


def build_nutrition(row: dict) -> dict:
    nutr = {}
    for val_col, unit_col, name in NUTRIENT_COLS:
        val = row.get(val_col)
        if val is not None and str(val) not in ("nan","None",""):
            try:
                nutr[name] = {"value": round(float(val), 3),
                              "unit": row.get(unit_col) or "g"}
            except: pass
    return nutr

def parse_list(v) -> list:
    if not v or str(v) in ("nan","None",""): return []
    return [x.strip() for x in str(v).split("|") if x.strip()]

def load_caches():
    client = get_bq()
    if not client: return
    try:
        query_rows = list(client.query(f"""
            SELECT * FROM `{PROJECT}.{DATASET}.{TABLE}`
            ORDER BY brand, product_name
        """).result())
        rows = [clean_row(dict(r)) for r in query_rows]
        # Derive and normalise image filenames from barcode
        for row in rows:
            derive_images(row)
        for row in rows:
            for field in ("back_image", "front_image", "product_image"):
                v = row.get(field)
                if v and str(v).endswith("_b.png"):
                    row[field] = v[:-6] + "_ba.png"
            # Fix image_files pipe list too
            imgs = row.get("image_files")
            if imgs:
                fixed = []
                for f in str(imgs).split("|"):
                    f = f.strip()
                    if f.endswith("_b.png"):
                        f = f[:-6] + "_ba.png"
                    if f:
                        fixed.append(f)
                row["image_files"] = " | ".join(fixed)
        set_cache("products", rows)
        print(f"[CACHE] {len(rows)} products")

        # Stats without pandas
        types  = {}
        brands = set()
        conf_total = 0
        lang_total = 0
        conf_count = 0
        for r in rows:
            pt = r.get("product_type") or "unknown"
            types[pt] = types.get(pt, 0) + 1
            if r.get("brand"): brands.add(r["brand"])
            if r.get("confidence"):
                conf_total += float(r["confidence"] or 0)
                conf_count += 1
            if r.get("language_count"):
                lang_total += float(r["language_count"] or 0)
        conf = (conf_total / conf_count * 100) if conf_count else 0
        langs = (lang_total / len(rows)) if rows else 0
        set_cache("stats", {
            "total_products": len(rows), "unique_brands": len(brands),
            "food":      types.get("food",0) + types.get("spice",0) + types.get("pickle",0),
            "cosmetic":  types.get("cosmetic",0),
            "household": types.get("household",0),
            "avg_confidence": round(conf, 1),
            "avg_languages":  round(langs, 1),
            "product_types":  {k: int(v) for k, v in types.items()},
        })

        # Regions
        counts = {}
        for row in rows:
            for r in parse_list(row.get("regions")):
                counts[r] = counts.get(r, 0) + 1
        region_codes_map = {
            "Tamil Nadu":"IN-TN","Puducherry":"IN-PY","Andhra Pradesh":"IN-AP",
            "Telangana":"IN-TG","Karnataka":"IN-KA","Kerala":"IN-KL",
            "Maharashtra":"IN-MH","Rajasthan":"IN-RJ","UP":"IN-UP","Bihar":"IN-BR",
            "West Bengal":"IN-WB","Tripura":"IN-TR","Gujarat":"IN-GJ",
            "Punjab":"IN-PB","Odisha":"IN-OD","Pan India":"IN",
            "Pan India (Muslim market)":"IN",
        }
        regions = [{"region": k, "region_code": region_codes_map.get(k, ""), "product_count": v}
                   for k, v in sorted(counts.items(), key=lambda x: -x[1])
                   if k not in ("Pan India (Muslim market)",)]
        set_cache("regions", regions)
        print(f"[CACHE] Stats and {len(regions)} regions loaded")
    except Exception as e:
        print(f"[CACHE ERROR] {e}"); traceback.print_exc()

def refresh_loop():
    while True:
        time.sleep(CACHE_TTL)
        load_caches()

threading.Thread(target=refresh_loop, daemon=True).start()


# ── Static files ───────────────────────────────────────────────────────────────
@app.get("/")
async def root():
    try:
        with open("index.html") as f: return HTMLResponse(f.read())
    except: return HTMLResponse("<h1>GS1 DataKart</h1>")

@app.get("/acquink_logo.png")
async def logo():
    return FileResponse("acquink_logo.png", media_type="image/png")

@app.get("/api/debug/product/{id}")
def debug_product(id: str):
    """Debug: show exactly what the cache has for this id."""
    items = get_cache("products") or []
    results = []
    for p in items:
        bc  = str(p.get("barcode") or "").strip()
        acq = str(p.get("acquink_id") or "").strip()
        if id.lower() in bc.lower() or id.lower() in acq.lower() or            id.lower() in (p.get("brand") or "").lower():
            results.append({
                "barcode": bc,
                "acquink_id": acq,
                "brand": p.get("brand"),
                "product_name": p.get("product_name"),
                "product_type": p.get("product_type"),
                "front_image": p.get("front_image"),
            })
    return {
        "query": id,
        "cache_size": len(items),
        "matches": results,
        "cache_loaded": cache_valid("products"),
    }

@app.get("/api/debug/cache")
def debug_cache():
    """Debug: show cache stats and first 5 products."""
    items = get_cache("products") or []
    return {
        "cache_size": len(items),
        "cache_loaded": cache_valid("products"),
        "sample": [
            {"barcode": p.get("barcode"), "acquink_id": p.get("acquink_id"),
             "brand": p.get("brand"), "product_type": p.get("product_type")}
            for p in items[:5]
        ]
    }

@app.get("/health")
@app.get("/api/health")
async def health():
    return {"status": "healthy" if get_bq() else "degraded",
            "bq": bool(get_bq()), "gemini": bool(get_gemini()),
            "cache": {k: cache_valid(k) for k in CACHE}}


# ── Data APIs ──────────────────────────────────────────────────────────────────
@app.get("/api/stats")
def get_stats():
    return get_cache("stats") or {"total_products": 0, "unique_brands": 0}

@app.get("/api/regions")
def get_regions():
    return get_cache("regions") or []

@app.get("/api/search")
def search(q: str = "", product_type: str = "", region_code: str = "", limit: int = 0):
    items = get_cache("products") or []
    if q:
        ql = q.lower()
        items = [p for p in items if
                 ql in (p.get("brand") or "").lower() or
                 ql in (p.get("product_name") or "").lower() or
                 (str(p.get("barcode") or "") == q and str(p.get("barcode") or "") not in ("nan","None","","<NA>")) or
                 (p.get("acquink_id") or "").lower() == ql]
    if product_type:
        items = [p for p in items
                 if (p.get("product_type") or "").lower() == product_type.lower()]
    if region_code:
        items = [p for p in items
                 if region_code.upper() in (p.get("region_codes") or "")]
    return items[:limit] if limit > 0 else items

@app.get("/api/autocomplete")
def autocomplete(q: str = ""):
    if len(q) < 2: return []
    ql = q.lower()
    items, seen, results = get_cache("products") or [], set(), []
    for p in items:
        brand = p.get("brand") or ""
        name  = p.get("product_name") or ""
        if ql in brand.lower() or ql in name.lower():
            key = brand
            if key not in seen:
                seen.add(key)
                results.append({
                    "label": f"{brand} — {name}",
                    "barcode": p.get("barcode") or p.get("acquink_id"),
                    "product_type": p.get("product_type"),
                })
        if len(results) >= 8: break
    return results

@app.get("/api/product/{id}")
def get_product(id: str):
    if not id or id in ("nan","None","<NA>",""):
        raise HTTPException(404, "Invalid product ID")
    items = get_cache("products") or []
    for p in items:
        bc  = str(p.get("barcode") or "").strip()
        acq = str(p.get("acquink_id") or "").strip()
        # Normalise — treat <NA> as empty
        if bc in ("nan","None","<NA>",""): bc = ""
        if acq in ("nan","None","<NA>",""): acq = ""
        barcode_match = bc == id
        acquink_match = acq == id
        if barcode_match or acquink_match:
            result = dict(p)
            p = derive_images(p)
            result["nutrition"]     = build_nutrition(p)
            result["ingredients"]   = parse_list(p.get("ingredients"))
            result["languages"]     = parse_list(p.get("languages"))
            result["language_codes"]= parse_list(p.get("language_codes"))
            result["regions"]       = parse_list(p.get("regions"))
            result["region_codes"]  = parse_list(p.get("region_codes"))
            result["certifications"]= parse_list(p.get("certifications"))
            result["allergens"]     = parse_list(p.get("allergens"))
            result["flags"]         = parse_list(p.get("flags"))
            return result
    raise HTTPException(404, "Product not found")

@app.get("/api/suggestions")
def get_suggestions():
    return [
        "Which food products have the highest sodium content?",
        "Compare Dhurka and TORAN on nutrition",
        "Which brands cover more than 6 Indian states?",
        "Show all products with HALAL certification",
        "Which products are missing FSSAI numbers?",
        "What is the average energy content of food products?",
        "Which brands target Tamil Nadu and Karnataka?",
        "Show products with more than 10 ingredients",
        "Compare cosmetic brands by regional reach",
        "Which products have allergen warnings?",
    ]


# ── Category personas ────────────────────────────────────────────────────────
CATEGORY_PERSONAS = {
    "food":      "You are a nutrition and food safety specialist.",
    "spice":     "You are a nutrition and food safety specialist.",
    "pickle":    "You are a nutrition and food safety specialist.",
    "beverage":  "You are a nutrition and food safety specialist.",
    "cosmetic":  "You are a cosmetic ingredient safety specialist.",
    "household": "You are a consumer product safety specialist.",
    "cleaning":  "You are a consumer product safety specialist.",
    "pharma":    "You are a healthcare product specialist.",
    "medical_device": "You are a healthcare product specialist.",
    "toy":       "You are a children product safety specialist.",
    "incense":   "You are a consumer product safety specialist.",
}

INTELLIGENCE_TEMPLATE = """{persona}

Analyse this product label data and provide structured intelligence.
Return your response in this exact format:

**Pros**
- [key positive point]
- [key positive point]

**Cautions**
- [key caution or concern]
- [key caution or concern]

**Target Consumer**
[one sentence on ideal consumer segment]

**Verdict**
[one sentence summary]

Product data:
Brand: {brand}
Product: {product_name}
Type: {product_type}
FSSAI: {fssai}
Certifications: {certifications}
Ingredients ({ingredient_count}): {ingredients}
Nutrition per 100g: {nutrition}
Label languages: {languages}
Regions targeted: {regions}
Manufacturer: {manufacturer_name}

Be specific. Use only the data provided. Keep each point under 15 words."""


class IntelligenceRequest(BaseModel):
    product_id: str

@app.post("/api/intelligence")
async def get_intelligence(req: IntelligenceRequest):
    """Generate and cache product intelligence via the BQ agent."""
    # Check cache first
    client = get_bq()
    if client:
        try:
            from google.cloud.bigquery import QueryJobConfig, ScalarQueryParameter
            cfg = QueryJobConfig(query_parameters=[
                ScalarQueryParameter("id","STRING",req.product_id)])
            q = f"""SELECT intelligence_text FROM `{PROJECT}.{DATASET}.{TABLE}`
                    WHERE barcode=@id OR acquink_id=@id LIMIT 1"""
            rows = list(client.query(q, job_config=cfg).result())
            if rows:
                cached = dict(rows[0]).get("intelligence_text")
                if cached and str(cached) not in ("nan","None",""):
                    return {"intelligence": cached, "cached": True}
        except Exception as e:
            print(f"[INTEL CACHE CHECK] {e}")

    # Get product data
    try:
        p = get_product(req.product_id)
    except:
        raise HTTPException(404, "Product not found")

    pt       = (p.get("product_type") or "unknown").lower()
    persona  = CATEGORY_PERSONAS.get(pt, "You are a consumer product specialist.")
    nutr_str = ", ".join(f"{k}: {v['value']} {v['unit']}"
                         for k,v in (p.get("nutrition") or {}).items()) or "Not available"
    ing_list = (p.get("ingredients") or [])

    prompt = INTELLIGENCE_TEMPLATE.format(
        persona        = persona,
        brand          = p.get("brand") or "Unknown",
        product_name   = p.get("product_name") or "",
        product_type   = pt,
        fssai          = p.get("fssai") or "Not found",
        certifications = ", ".join(p.get("certifications") or []) or "None",
        ingredients    = ", ".join(ing_list[:15]) or "Not available",
        ingredient_count = len(ing_list),
        nutrition      = nutr_str,
        languages      = ", ".join(p.get("languages") or []) or "Not detected",
        regions        = ", ".join((p.get("regions") or [])[:8]) or "Not mapped",
        manufacturer_name = p.get("manufacturer_name") or "Not found",
    )

    intelligence = ""

    # Call same Vertex agent
    try:
        from google.cloud import geminidataanalytics_v1alpha as gda
        cc = get_data_chat_client()
        if cc:
            parent     = f"projects/{PROJECT}/locations/{AGENT_LOCATION}"
            agent_path = f"{parent}/dataAgents/{AGENT_ID}"
            conv_id    = f"intel-{req.product_id}-{uuid.uuid4().hex[:6]}"
            conv_path  = cc.conversation_path(PROJECT, AGENT_LOCATION, conv_id)
            try:
                cc.create_conversation(request=gda.CreateConversationRequest(
                    parent=parent, conversation_id=conv_id,
                    conversation=gda.Conversation(agents=[agent_path])
                ))
            except: pass
            stream = cc.chat(request={
                "parent": parent,
                "conversation_reference": {
                    "conversation": conv_path,
                    "data_agent_context": {"data_agent": agent_path}
                },
                "messages": [{"user_message": {"text": prompt}}]
            })
            for chunk in stream:
                if hasattr(chunk,"agent_message") and hasattr(chunk.agent_message,"text"):
                    for part in chunk.agent_message.text.parts:
                        intelligence += str(part)
    except Exception as e:
        print(f"[INTEL AGENT] {e}")

    # Fallback to Gemini direct
    if not intelligence:
        model = get_gemini()
        if model:
            try:
                resp = model.generate_content(prompt)
                intelligence = resp.text or ""
            except Exception as e:
                intelligence = f"Intelligence generation failed: {str(e)[:100]}"

    # Cache in BigQuery
    if intelligence and client:
        try:
            safe = intelligence.replace("'","''")
            pid  = req.product_id.replace("'","''")
            upd  = f"""UPDATE `{PROJECT}.{DATASET}.{TABLE}`
                       SET intelligence_text = '{safe}'
                       WHERE barcode='{pid}' OR acquink_id='{pid}'"""
            client.query(upd).result()
            print(f"[INTEL] Cached for {pid}")
        except Exception as e:
            print(f"[INTEL CACHE WRITE] {e}")

    return {"intelligence": intelligence, "cached": False}


@app.get("/api/accuracy")
def get_accuracy():
    """Return accuracy stats at product, category and overall level."""
    client = get_bq()
    if not client:
        return {}
    try:
        q = f"""
        SELECT
            product_type,
            COUNT(*) as count,
            ROUND(AVG(CASE WHEN confidence IS NOT NULL THEN confidence END)*100,1) as avg_conf,
            COUNTIF(fssai IS NOT NULL AND LENGTH(fssai)>=14) as with_fssai,
            COUNTIF(brand IS NOT NULL) as with_brand,
            COUNTIF(languages IS NOT NULL) as with_languages
        FROM `{PROJECT}.{DATASET}.{TABLE}`
        GROUP BY product_type
        ORDER BY count DESC
        """
        rows = list(client.query(q).result())
        return [dict(r) for r in rows]
    except Exception as e:
        return {"error": str(e)}


# ── Chat API (same pattern as Hotels) ─────────────────────────────────────────
class ChatRequest(BaseModel):
    message: str
    product_id: Optional[str] = None
    conversation_id: Optional[str] = None

@app.post("/api/chat")
async def chat(request: ChatRequest):
    system_prompt = get_agent_prompt("gs1")
    stats         = get_cache("stats") or {}

    # Build data context — same pattern as Hotels
    if request.product_id:
        try:
            p      = get_product(request.product_id)
            langs  = p.get("languages") or []
            codes  = p.get("language_codes") or []
            regions= p.get("regions") or []
            nutr   = p.get("nutrition") or {}
            ing    = p.get("ingredients") or []
            conf   = round((p.get("confidence") or 0) * 100)

            lang_lines = "\n".join(
                f"  {l} ({codes[i] if i < len(codes) else '?'}) "
                f"→ {', '.join([r for r in regions if r])[:60]}"
                for i, l in enumerate(langs[:9])
            ) or "  No language data"

            nutr_lines = "\n".join(
                f"  {k}: {v['value']} {v['unit']}" for k, v in nutr.items()
            ) or "  No nutrition data on this panel"

            ing_lines = ", ".join(ing[:20]) if ing else "Not available on this panel"

            front_url = f"{GCS_BASE}/{p['front_image']}" if p.get("front_image") else "N/A"
            back_url  = f"{GCS_BASE}/{p['back_image']}"  if p.get("back_image")  else "N/A"

            data_context = f"""
=== PRODUCT INTELLIGENCE CONTEXT ===
Brand: {p.get('brand') or 'Unknown'} | Product: {p.get('product_name') or 'Unknown'}
Type: {p.get('product_type') or 'unknown'} | Barcode: {p.get('barcode') or p.get('acquink_id')}
FSSAI: {p.get('fssai') or 'NOT FOUND — compliance gap'} | MRP: ₹{p.get('mrp') or 'N/A'}
Net Weight: {p.get('net_weight') or 'N/A'} | Confidence: {conf}%
Barcode Verified: {p.get('barcode_verified')} | Certifications: {', '.join(p.get('certifications') or [])}
Manufacturer: {p.get('manufacturer_name') or 'N/A'}
Address: {p.get('manufacturer_address') or 'N/A'}
Contact: {p.get('complaint_number') or 'N/A'} | {p.get('complaint_email') or 'N/A'}
Front image: {front_url}
Back image:  {back_url}

=== LABEL LANGUAGES & REGIONAL REACH ===
{lang_lines}

=== NUTRITION (per 100g) ===
{nutr_lines}

=== INGREDIENTS ({p.get('ingredient_count') or len(ing)}) ===
{ing_lines}

INSTRUCTION: Use ONLY the above data for THIS product.
For cross-product questions (e.g. "which product has highest sodium") — query BigQuery directly.
Table: gen-lang-client-0143536012.gs1_datakart.products
Key nutrition columns: n_sodium, n_energy, n_protein, n_total_fat, n_total_carbohydrate
Never say "broader dataset required" — you have full BigQuery access."""
        except Exception as e:
            data_context = f"No product context. Dataset: {stats.get('total_products',0)} products indexed."
    else:
        data_context = f"""
=== GS1 DATAKART DATASET ===
{stats.get('total_products',0)} products | {stats.get('unique_brands',0)} brands
Food: {stats.get('food',0)} | Cosmetic: {stats.get('cosmetic',0)} | Household: {stats.get('household',0)}
Avg confidence: {stats.get('avg_confidence',0)}%

BigQuery table: gen-lang-client-0143536012.gs1_datakart.products
Key columns: barcode, acquink_id, brand, product_name, product_type,
             fssai, mrp, net_weight, manufacturer_name,
             n_energy, n_protein, n_total_carbohydrate, n_total_fat,
             n_sodium, n_total_sugars, n_dietary_fiber, n_saturated_fat,
             ingredients, ingredient_count, languages, language_count,
             regions, certifications, confidence, intelligence_text,
             front_image, back_image

INSTRUCTION: You have DIRECT access to this BigQuery table. For any question about
multiple products, comparisons, rankings, or statistics — run a SQL query against
gen-lang-client-0143536012.gs1_datakart.products and return real data.
Never say "broader dataset required" — you already have it. Query it."""

    # ── Pre-execute BQ for cross-product questions ──────────────────────────
    bq_results_text = ""
    cross_product_keywords = [
        "highest","lowest","most","least","which product","which brand",
        "compare","ranking","rank","top","bottom","all products","all food",
        "across","average","sodium","sugar","calories","energy","protein",
        "fat","ingredients","FSSAI","missing","compliance","how many",
        "list all","show all","what products"
    ]
    is_cross_product = any(kw.lower() in request.message.lower() for kw in cross_product_keywords)

    if is_cross_product and not request.product_id:
        try:
            bq = get_bq()
            if bq:
                # Build a targeted query based on question intent
                msg_lower = request.message.lower()
                if any(x in msg_lower for x in ["sodium","salt"]):
                    q = """SELECT brand, product_name, product_type,
                        CAST(n_sodium AS STRING) AS n_sodium, n_sodium_unit
                        FROM `gen-lang-client-0143536012.gs1_datakart.products`
                        WHERE n_sodium IS NOT NULL AND product_type IN ('food','spice','pickle','beverage')
                        ORDER BY n_sodium DESC LIMIT 10"""
                elif any(x in msg_lower for x in ["calori","energy","kcal"]):
                    q = """SELECT brand, product_name, product_type,
                        CAST(n_energy AS STRING) AS n_energy, n_energy_unit
                        FROM `gen-lang-client-0143536012.gs1_datakart.products`
                        WHERE n_energy IS NOT NULL ORDER BY n_energy DESC LIMIT 10"""
                elif any(x in msg_lower for x in ["sugar"]):
                    q = """SELECT brand, product_name,
                        CAST(n_total_sugars AS STRING) AS n_total_sugars, n_total_sugars_unit
                        FROM `gen-lang-client-0143536012.gs1_datakart.products`
                        WHERE n_total_sugars IS NOT NULL ORDER BY n_total_sugars DESC LIMIT 10"""
                elif any(x in msg_lower for x in ["protein"]):
                    q = """SELECT brand, product_name,
                        CAST(n_protein AS STRING) AS n_protein, n_protein_unit
                        FROM `gen-lang-client-0143536012.gs1_datakart.products`
                        WHERE n_protein IS NOT NULL ORDER BY n_protein DESC LIMIT 10"""
                elif any(x in msg_lower for x in ["fat"]):
                    q = """SELECT brand, product_name,
                        CAST(n_total_fat AS STRING) AS n_total_fat, n_total_fat_unit
                        FROM `gen-lang-client-0143536012.gs1_datakart.products`
                        WHERE n_total_fat IS NOT NULL ORDER BY n_total_fat DESC LIMIT 10"""
                elif any(x in msg_lower for x in ["fssai","compliance","certified","missing"]):
                    q = """SELECT brand, product_name, product_type, fssai,
                        CASE WHEN fssai IS NULL THEN 'MISSING' ELSE 'OK' END AS fssai_status
                        FROM `gen-lang-client-0143536012.gs1_datakart.products`
                        WHERE product_type IN ('food','spice','pickle','beverage','cosmetic','pharma')
                        ORDER BY fssai_status DESC, brand LIMIT 20"""
                elif any(x in msg_lower for x in ["ingredient","how many ingredient"]):
                    q = """SELECT brand, product_name, product_type, ingredient_count, ingredients
                        FROM `gen-lang-client-0143536012.gs1_datakart.products`
                        WHERE ingredient_count IS NOT NULL ORDER BY ingredient_count DESC LIMIT 10"""
                else:
                    q = """SELECT brand, product_name, product_type, mrp, net_weight, fssai,
                        n_energy, n_sodium, n_total_fat, ingredient_count, confidence
                        FROM `gen-lang-client-0143536012.gs1_datakart.products`
                        ORDER BY brand, product_name LIMIT 30"""

                rows = list(bq.query(q).result())
                if rows:
                    lines = []
                    for r in rows:
                        lines.append(" | ".join(f"{k}: {v}" for k, v in dict(r).items() if v is not None))
                    bq_results_text = "\n=== LIVE BIGQUERY DATA ===\n" + "\n".join(lines)
                    print(f"[CHAT] BQ pre-query: {len(rows)} rows")
        except Exception as e:
            print(f"[CHAT] BQ pre-query failed: {e}")

    full_prompt = f"""{system_prompt}

{data_context}
{bq_results_text}

=== USER QUERY ===
{request.message}

CRITICAL RULES FOR YOUR RESPONSE:
1. Answer in plain conversational English. Use bullet points or a simple table if helpful.
2. NEVER show SQL code. NEVER use code blocks. The user is a business person, not a developer.
3. Use the LIVE BIGQUERY DATA above to answer with real product names and numbers.
4. If data shows unusual values (e.g. sodium 3.4g which seems high), flag it as a possible OCR issue.
5. Be concise — 3 to 8 lines maximum unless a list is needed.
6. Never say you cannot access the database. The data is already provided above."""

    conv_id       = request.conversation_id or f"gs1-{uuid.uuid4().hex[:8]}"
    response_text = ""

    # Path 1: geminidataanalytics agent (same as Hotels)
    try:
        from google.cloud import geminidataanalytics_v1alpha as gda
        cc = get_data_chat_client()
        if cc:
            parent     = f"projects/{PROJECT}/locations/{AGENT_LOCATION}"
            agent_path = f"{parent}/dataAgents/{AGENT_ID}"
            conv_path  = cc.conversation_path(PROJECT, AGENT_LOCATION, conv_id)
            try:
                cc.get_conversation(name=conv_path)
            except:
                cc.create_conversation(request=gda.CreateConversationRequest(
                    parent=parent, conversation_id=conv_id,
                    conversation=gda.Conversation(agents=[agent_path])
                ))
            stream = cc.chat(request={
                "parent": parent,
                "conversation_reference": {
                    "conversation": conv_path,
                    "data_agent_context": {"data_agent": agent_path}
                },
                "messages": [{"user_message": {"text": full_prompt}}]
            })
            for chunk in stream:
                if hasattr(chunk, "agent_message") and hasattr(chunk.agent_message, "text"):
                    for part in chunk.agent_message.text.parts:
                        response_text += str(part)
            print(f"[CHAT] Agent response: {len(response_text)} chars")
    except Exception as e:
        print(f"[CHAT] Agent failed: {e} — using Gemini fallback")

    # Path 2: GenerativeModel fallback (same as Hotels)
    if not response_text:
        model = get_gemini()
        if model:
            try:
                resp = model.generate_content(full_prompt)
                response_text = resp.text or "No response generated."
            except Exception as e:
                response_text = f"Error generating response: {str(e)}"
        else:
            response_text = "Chat service unavailable. Check GCP_CREDENTIALS_JSON."

    return {"response": response_text, "conversation_id": conv_id}


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8080)
