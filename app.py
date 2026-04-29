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

@app.get("/health")
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
def search(q: str = "", product_type: str = "", region_code: str = "", limit: int = 60):
    items = get_cache("products") or []
    if q:
        ql = q.lower()
        items = [p for p in items if
                 ql in (p.get("brand") or "").lower() or
                 ql in (p.get("product_name") or "").lower() or
                 str(p.get("barcode") or "") == q or
                 (p.get("acquink_id") or "").lower() == ql]
    if product_type:
        items = [p for p in items
                 if (p.get("product_type") or "").lower() == product_type.lower()]
    if region_code:
        items = [p for p in items
                 if region_code.upper() in (p.get("region_codes") or "")]
    return items[:limit]

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
    items = get_cache("products") or []
    for p in items:
        if str(p.get("barcode") or "") == id or \
           str(p.get("acquink_id") or "") == id:
            result = dict(p)
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

INSTRUCTION: Use ONLY the exact data above. Never invent numbers.
If barcode_verified=False → note it needs GS1 verification.
Include image URLs when showing this product."""
        except Exception as e:
            data_context = f"No product context. Dataset: {stats.get('total_products',0)} products indexed."
    else:
        data_context = f"""
=== GS1 DATAKART DATASET ===
{stats.get('total_products',0)} products | {stats.get('unique_brands',0)} brands
Food: {stats.get('food',0)} | Cosmetic: {stats.get('cosmetic',0)} | Household: {stats.get('household',0)}
Avg confidence: {stats.get('avg_confidence',0)}%
BigQuery: gen-lang-client-0143536012.gs1_datakart.products
Images: https://storage.googleapis.com/gs1-datakart-images/

INSTRUCTION: Query the BigQuery table to answer accurately. Never fabricate data."""

    full_prompt = f"""{system_prompt}

{data_context}

=== USER QUERY ===
{request.message}

Remember: Use ONLY the data above or query BigQuery. Never invent numbers."""

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
