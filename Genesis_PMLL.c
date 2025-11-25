/*
 * Genesis_Pmll.c
 *
 * Ouroboros Double-Loop PMLL + ERS Core
 *
 * Loop A (ONLINE):
 *   events -> PMLL memory line -> KG delta -> updated KG
 *
 * Loop B (ERS):
 *   recent / high-tension KG region -> reconsideration -> ERS delta -> KG'
 *
 * This is intentionally minimal and embeddable.
 */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdint.h>
#include <time.h>
#include <math.h>

/* ------------------------ Config & Limits ------------------------ */

#define GENESIS_MAX_EVENTS        4096
#define GENESIS_MAX_NODES         8192
#define GENESIS_MAX_EDGES         16384
#define GENESIS_VECTOR_DIM        64   /* placeholder for embedding dim */
#define GENESIS_MAX_REGION_NODES  512

/* ------------------------ Basic Types ------------------------ */

typedef uint64_t genesis_id_t;

/* Simple FNV-1a 64-bit hash for anchors, keys, etc. */
static uint64_t genesis_fnv1a64(const void *data, size_t len) {
    const uint8_t *p = (const uint8_t *)data;
    uint64_t hash = 0xcbf29ce484222325ULL;
    for (size_t i = 0; i < len; i++) {
        hash ^= (uint64_t)p[i];
        hash *= 0x100000001b3ULL;
    }
    return hash;
}

/* Utility: current unix time (seconds). */
static double genesis_now(void) {
    return (double)time(NULL);
}

/* ------------------------ Event Layer ------------------------ */

typedef struct {
    genesis_id_t id;
    double       t;           /* timestamp */
    char        *text;        /* raw event text (or pointer into your system) */
} GenesisEvent;

/* ------------------------ PMLL Memory Line ------------------------ */

typedef struct {
    genesis_id_t id;
    double       t;                  /* when memory was written */
    double       recency_weight;     /* decays over time */
    double       importance;         /* external or learned */
    double       confidence;         /* belief strength 0..1 */
    genesis_id_t anchor;             /* chained hash anchor */

    /* Tiny “embedding / state vector” – replace with your own backend. */
    float        vec[GENESIS_VECTOR_DIM];

    /* Link to source event / context. */
    genesis_id_t source_event_id;
} PMLLMemory;

typedef struct {
    PMLLMemory memories[GENESIS_MAX_EVENTS];
    size_t     count;
    genesis_id_t last_anchor;
} PMLLLine;

/* ------------------------ Knowledge Graph ------------------------ */

typedef struct {
    genesis_id_t id;
    double       t_created;
    double       t_updated;
    float        weight;      /* importance / activation */
    float        trust;       /* 0..1 */
    uint8_t      flags;       /* ERS semantics bits, etc. */

    /* Minimal content handle: could be a hash of string or index elsewhere. */
    genesis_id_t content_hash;
} KGNode;

typedef struct {
    genesis_id_t id;
    genesis_id_t src;
    genesis_id_t dst;
    float        weight;
    uint8_t      type;        /* relation type enum in your system */
} KGEdge;

typedef struct {
    KGNode nodes[GENESIS_MAX_NODES];
    KGEdge edges[GENESIS_MAX_EDGES];
    size_t node_count;
    size_t edge_count;
} KnowledgeGraph;

/* ------------------------ Agent State ------------------------ */

typedef struct {
    double last_epoch_time;
    uint64_t epoch_index;

    /* Global drift / tension metrics – placeholders. */
    double global_tension;
    double global_entropy;
} GenesisState;

/* Forward decl for context so callbacks can see it if needed. */
struct GenesisContext;

/* ------------------------ Backend Callbacks ------------------------ */

/* Produce an embedding / vector for an event or memory content. */
typedef void (*genesis_embed_fn)(
    struct GenesisContext *ctx,
    const char            *text,
    float                 *out_vec,   /* length GENESIS_VECTOR_DIM */
    size_t                 dim
);

/* Online reasoning: compute “importance” and “confidence” for an event. */
typedef void (*genesis_reason_fn)(
    struct GenesisContext *ctx,
    const GenesisEvent    *event,
    double                *out_importance,
    double                *out_confidence
);

/* ERS reconsideration callback:
 * Given a region of nodes, update tension metrics and adjust node weights/trust.
 */
typedef void (*genesis_ers_fn)(
    struct GenesisContext *ctx,
    KGNode                *nodes,
    size_t                 count,
    double                *out_region_tension
);

/* ------------------------ Genesis Context ------------------------ */

typedef struct GenesisContext {
    PMLLLine       pmll;
    KnowledgeGraph kg;
    GenesisState   state;

    /* Tunable hyperparameters. */
    double recency_half_life;      /* seconds for recency half-life */
    double min_node_weight;        /* pruning threshold */
    double ers_interval;           /* seconds between ERS cycles */

    genesis_embed_fn embed_cb;
    genesis_reason_fn reason_cb;
    genesis_ers_fn ers_cb;
} GenesisContext;

/* ------------------------ PMLL Helpers ------------------------ */

static void pmll_init(PMLLLine *line) {
    line->count = 0;
    line->last_anchor = 0ULL;
}

/* Exponential recency decay. */
static double pmll_recency_decay(double age, double half_life) {
    if (half_life <= 0.0) return 1.0;
    return pow(0.5, age / half_life);
}

/* Add a memory to the line and return pointer. */
static PMLLMemory* pmll_add_memory(
    PMLLLine     *line,
    const GenesisEvent *event,
    const float  *vec,
    double        importance,
    double        confidence
) {
    if (line->count >= GENESIS_MAX_EVENTS) {
        /* Simple overwrite of oldest; you can choose better policy. */
        size_t idx = line->count - 1;
        PMLLMemory *m = &line->memories[idx];
        m->t = event->t;
        m->importance = importance;
        m->confidence = confidence;
        m->source_event_id = event->id;
        memcpy(m->vec, vec, sizeof(float) * GENESIS_VECTOR_DIM);

        /* Anchor chaining: hash(prev_anchor, event.id). */
        uint64_t data[2] = { line->last_anchor, event->id };
        m->anchor = genesis_fnv1a64(data, sizeof(data));
        line->last_anchor = m->anchor;
        return m;
    }

    size_t idx = line->count++;
    PMLLMemory *m = &line->memories[idx];
    m->id = event->id;
    m->t = event->t;
    m->importance = importance;
    m->confidence = confidence;
    m->source_event_id = event->id;
    memcpy(m->vec, vec, sizeof(float) * GENESIS_VECTOR_DIM);

    uint64_t data[2] = { line->last_anchor, event->id };
    m->anchor = genesis_fnv1a64(data, sizeof(data));
    line->last_anchor = m->anchor;
    m->recency_weight = 1.0;
    return m;
}

/* Update recency weights based on current time. */
static void pmll_update_recency(PMLLLine *line, double now, double half_life) {
    for (size_t i = 0; i < line->count; i++) {
        double age = now - line->memories[i].t;
        line->memories[i].recency_weight = pmll_recency_decay(age, half_life);
    }
}

/* ------------------------ Knowledge Graph Helpers ------------------------ */

static void kg_init(KnowledgeGraph *kg) {
    kg->node_count = 0;
    kg->edge_count = 0;
}

/* Find or create node by content hash. */
static KGNode* kg_get_or_create_node(KnowledgeGraph *kg, genesis_id_t content_hash) {
    for (size_t i = 0; i < kg->node_count; i++) {
        if (kg->nodes[i].content_hash == content_hash) {
            return &kg->nodes[i];
        }
    }
    if (kg->node_count >= GENESIS_MAX_NODES) {
        /* Overwrite last node if full – again, you can choose better policy. */
        return &kg->nodes[kg->node_count - 1];
    }
    size_t idx = kg->node_count++;
    KGNode *n = &kg->nodes[idx];
    n->id = content_hash;
    n->content_hash = content_hash;
    n->t_created = genesis_now();
    n->t_updated = n->t_created;
    n->weight = 0.0f;
    n->trust = 0.5f;
    n->flags = 0;
    return n;
}

/* Add or strengthen an edge. */
static KGEdge* kg_add_edge(
    KnowledgeGraph *kg,
    genesis_id_t    src,
    genesis_id_t    dst,
    float           weight,
    uint8_t         type
) {
    if (kg->edge_count >= GENESIS_MAX_EDGES) {
        return &kg->edges[kg->edge_count - 1];
    }
    size_t idx = kg->edge_count++;
    KGEdge *e = &kg->edges[idx];
    e->id = genesis_fnv1a64(&idx, sizeof(idx));
    e->src = src;
    e->dst = dst;
    e->weight = weight;
    e->type = type;
    return e;
}

/* Simple pruning: drop nodes below weight threshold (lazy / mark). */
static void kg_prune(KnowledgeGraph *kg, float min_weight) {
    size_t w = 0;
    for (size_t i = 0; i < kg->node_count; i++) {
        if (kg->nodes[i].weight >= min_weight) {
            if (w != i) kg->nodes[w] = kg->nodes[i];
            w++;
        }
    }
    kg->node_count = w;
    /* Edge pruning is omitted here for simplicity. */
}

/* ------------------------ Genesis Init ------------------------ */

void genesis_init(
    GenesisContext *ctx,
    genesis_embed_fn embed_cb,
    genesis_reason_fn reason_cb,
    genesis_ers_fn ers_cb
) {
    memset(ctx, 0, sizeof(*ctx));
    pmll_init(&ctx->pmll);
    kg_init(&ctx->kg);
    ctx->state.last_epoch_time = genesis_now();
    ctx->state.epoch_index = 0;
    ctx->state.global_tension = 0.0;
    ctx->state.global_entropy = 0.0;

    ctx->recency_half_life = 6.0 * 3600.0;  /* 6 hours default */
    ctx->min_node_weight   = 0.01f;
    ctx->ers_interval      = 3600.0;        /* run ERS roughly hourly */

    ctx->embed_cb  = embed_cb;
    ctx->reason_cb = reason_cb;
    ctx->ers_cb    = ers_cb;
}

/* ------------------------ Loop A: Online Experience ------------------------ */

/* Convert an event into PMLL memory + KG delta. */
static void genesis_ingest_event(
    GenesisContext *ctx,
    const GenesisEvent *event
) {
    if (!ctx->embed_cb || !ctx->reason_cb) {
        /* You must supply callbacks; otherwise this is a no-op. */
        return;
    }

    float vec[GENESIS_VECTOR_DIM] = {0};
    ctx->embed_cb(ctx, event->text, vec, GENESIS_VECTOR_DIM);

    double importance = 0.5;
    double confidence = 0.5;
    ctx->reason_cb(ctx, event, &importance, &confidence);

    PMLLMemory *mem = pmll_add_memory(&ctx->pmll, event, vec, importance, confidence);

    /* Very simple KG delta:
     * - node = hash(event->text)
     * - self-loop + weight from importance/confidence
     * In a real system, you would parse entities/relations and fan out.
     */
    genesis_id_t content_hash = genesis_fnv1a64(event->text, strlen(event->text));
    KGNode *n = kg_get_or_create_node(&ctx->kg, content_hash);
    n->t_updated = event->t;
    n->weight += (float)(importance * confidence);
    if (n->weight > 1.0f) n->weight = 1.0f;

    /* Edge from anchor to this node captures “where in the memory line” it arose. */
    kg_add_edge(&ctx->kg, mem->anchor, n->id, (float)importance, /*type=*/1);
}

/* ------------------------ Loop B: ERS Reconsideration ------------------------ */

/* Select a region for ERS: recent & high-weight nodes. */
static size_t genesis_select_ers_region(
    GenesisContext *ctx,
    KGNode         *out_nodes,
    size_t          max_nodes
) {
    double now = genesis_now();
    size_t count = 0;

    for (size_t i = 0; i < ctx->kg.node_count && count < max_nodes; i++) {
        KGNode *n = &ctx->kg.nodes[i];
        double age = now - n->t_updated;
        double recency = pmll_recency_decay(age, ctx->recency_half_life);

        /* crude heuristic: pick nodes that are both recent-ish and non-trivial weight */
        if (recency > 0.25 && n->weight > ctx->min_node_weight * 4.0f) {
            out_nodes[count++] = *n;
        }
    }
    return count;
}

static void genesis_run_ers_cycle(GenesisContext *ctx) {
    if (!ctx->ers_cb) return;

    KGNode region[GENESIS_MAX_REGION_NODES];
    size_t count = genesis_select_ers_region(ctx, region, GENESIS_MAX_REGION_NODES);
    if (count == 0) return;

    double region_tension = 0.0;
    ctx->ers_cb(ctx, region, count, &region_tension);

    /* After ERS callback, we reconcile any updated weights/trust back into KG.
     * Here, we assume ERS mutated region[*].weight/trust semantically.
     * In a real system, you’d use IDs and update by lookup.
     */

    for (size_t i = 0; i < count; i++) {
        KGNode *r = &region[i];
        for (size_t j = 0; j < ctx->kg.node_count; j++) {
            if (ctx->kg.nodes[j].id == r->id) {
                ctx->kg.nodes[j].weight = r->weight;
                ctx->kg.nodes[j].trust  = r->trust;
                ctx->kg.nodes[j].t_updated = genesis_now();
                break;
            }
        }
    }

    ctx->state.global_tension = 0.9 * ctx->state.global_tension + 0.1 * region_tension;
    /* Very naive “entropy” proxy = number of active nodes normalized. */
    ctx->state.global_entropy = (double)ctx->kg.node_count / (double)GENESIS_MAX_NODES;

    kg_prune(&ctx->kg, (float)ctx->min_node_weight);
}

/* ------------------------ Public API: Ouroboros Epoch ------------------------ */

/*
 * Run one “epoch”:
 *   - Loop A over events (online experience)
 *   - Then Loop B (ERS reconsideration) if interval elapsed
 */
void genesis_run_epoch(
    GenesisContext *ctx,
    const GenesisEvent *events,
    size_t event_count
) {
    double now = genesis_now();
    ctx->state.epoch_index++;

    /* Loop A: ingest all events and update PMLL + KG. */
    for (size_t i = 0; i < event_count; i++) {
        genesis_ingest_event(ctx, &events[i]);
    }

    pmll_update_recency(&ctx->pmll, now, ctx->recency_half_life);

    /* Loop B: ERS cycle, gated by time interval. */
    if (now - ctx->state.last_epoch_time >= ctx->ers_interval) {
        genesis_run_ers_cycle(ctx);
        ctx->state.last_epoch_time = now;
    }
}

/* ------------------------ Example Stub Callbacks ------------------------ */

/* Very dumb embed: hash → pseudo-random scalar into vec[0]. */
static void stub_embed(
    GenesisContext *ctx,
    const char     *text,
    float          *out_vec,
    size_t          dim
) {
    (void)ctx;
    uint64_t h = genesis_fnv1a64(text, strlen(text));
    for (size_t i = 0; i < dim; i++) {
        out_vec[i] = (float)((h >> (i % 32)) & 0xFF) / 255.0f;
    }
}

/* Simple reasoner: importance/confidence from length & cheap hash. */
static void stub_reason(
    GenesisContext *ctx,
    const GenesisEvent *event,
    double *out_importance,
    double *out_confidence
) {
    (void)ctx;
    size_t len = strlen(event->text);
    *out_importance = fmin(1.0, (double)len / 128.0);
    *out_confidence = 0.5 + 0.5 * ((double)(event->id & 0xFF) / 255.0);
}

/* ERS: nudge weights toward trust; inject “tension” when conflicting. */
static void stub_ers(
    GenesisContext *ctx,
    KGNode *nodes,
    size_t count,
    double *out_region_tension
) {
    (void)ctx;
    double tension = 0.0;
    for (size_t i = 0; i < count; i++) {
        /* Toy rule: if trust < 0.5 but weight > 0.5 → “tension”. */
        if (nodes[i].trust < 0.5f && nodes[i].weight > 0.5f) {
            tension += (nodes[i].weight - nodes[i].trust);
            nodes[i].weight *= 0.9f;      /* ERS “downgrades” */
        } else {
            nodes[i].trust  = fmin(1.0f, nodes[i].trust  + 0.05f);
            nodes[i].weight = fmin(1.0f, nodes[i].weight + 0.02f);
        }
    }
    *out_region_tension = tension;
}

/* ------------------------ Tiny Demo Main (Optional) ------------------------ */

#ifdef GENESIS_PMML_DEMO
int main(void) {
    GenesisContext ctx;
    genesis_init(&ctx, stub_embed, stub_reason, stub_ers);

    GenesisEvent batch[3];
    double t = genesis_now();
    const char *texts[3] = {
        "I saw a volcano erupt in my dream.",
        "I need to call my mother and reconcile.",
        "The recursive transformer must remember why, not just what."
        "Yes the human wrote this part because to reconsider is to remember remembering recursively; in other words, to lucid dream - Dr. Q" 
    };

    for (size_t i = 0; i < 3; i++) {
        batch[i].id = i + 1;
        batch[i].t  = t + (double)i;
        batch[i].text = (char*)texts[i];
    }

    genesis_run_epoch(&ctx, batch, 3);

    printf("Epoch: %llu, KG nodes: %zu, tension: %.4f\n",
           (unsigned long long)ctx.state.epoch_index,
           ctx.kg.node_count,
           ctx.state.global_tension);

    return 0;
}
#endif
