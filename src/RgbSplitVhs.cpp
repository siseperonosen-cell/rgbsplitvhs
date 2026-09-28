// RgbSplitVhs - efecto de pantalla (aberracion cromatica radial + RGB split + VHS)
// para Minecraft Bedrock en LeviLauncher (Android, ARM64, OpenGL ES 3).
//
// Como funciona: al cargarse, un hilo espera a que exista libminecraftpe.so y
// reemplaza en su memoria los punteros a eglSwapBuffers por una funcion propia.
// Esa funcion copia el frame terminado, lo redibuja con un shader y despues
// llama a la funcion original. No usa offsets de Minecraft ni el SDK del launcher,
// asi que no depende de la version del juego.
//
// Ajustes: cambia los valores de "namespace cfg" y vuelve a compilar.

#include <EGL/egl.h>
#include <GLES3/gl3.h>
#include <android/log.h>
#include <dlfcn.h>
#include <link.h>
#include <pthread.h>
#include <sys/mman.h>
#include <time.h>
#include <unistd.h>

#include <cmath>
#include <cstdint>
#include <cstring>

#define TAG "RgbSplitVhs"
#define LOGI(...) __android_log_print(ANDROID_LOG_INFO, TAG, __VA_ARGS__)
#define LOGE(...) __android_log_print(ANDROID_LOG_ERROR, TAG, __VA_ARGS__)

// ------------------------------------------------------------------ ajustes
namespace cfg {
constexpr float kChroma    = 0.030f;  // separacion radial RGB (0 = apagado). Mas en los bordes.
constexpr float kSplitX    = 0.0015f; // separacion RGB horizontal uniforme (0 = apagado)
constexpr float kScanline  = 0.050f;  // fuerza de las lineas de escaneo VHS
constexpr float kNoise     = 0.120f;  // ruido de cinta
constexpr float kJitter    = 0.004f;  // temblor horizontal de lineas
constexpr float kVignette  = 0.350f;  // oscurecimiento de bordes
constexpr float kTintR     = 1.10f;   // tinte de color (1,1,1 = sin tinte)
constexpr float kTintG     = 1.05f;
constexpr float kTintB     = 0.85f;
}  // namespace cfg

// ------------------------------------------------------------------ shaders
static const char* kVertSrc = R"GLSL(#version 300 es
out vec2 vUV;
void main() {
    vec2 p = vec2((gl_VertexID == 1) ? 3.0 : -1.0, (gl_VertexID == 2) ? 3.0 : -1.0);
    vUV = p * 0.5 + 0.5;
    gl_Position = vec4(p, 0.0, 1.0);
}
)GLSL";

static const char* kFragSrc = R"GLSL(#version 300 es
precision highp float;
in vec2 vUV;
out vec4 fragColor;
uniform sampler2D uScene;
uniform float uTime;
uniform vec2  uRes;
uniform float uChroma;
uniform float uSplit;
uniform float uScan;
uniform float uNoise;
uniform float uJitter;
uniform float uVignette;
uniform vec3  uTint;

float rnd(vec2 p) {
    return fract(sin(dot(p, vec2(12.9898, 78.233))) * 43758.5453);
}

void main() {
    vec2 uv = vUV;

    // bandas de glitch ocasionales + temblor fino
    float t = floor(uTime * 10.0);
    float row = floor(uv.y * 48.0);
    float glitch = step(0.93, rnd(vec2(row, t)));
    uv.x += glitch * (rnd(vec2(t, row)) - 0.5) * uJitter * 4.0;
    uv.x += (rnd(vec2(floor(uv.y * uRes.y * 0.5), t)) - 0.5) * uJitter * 0.25;

    // aberracion cromatica: radial + horizontal
    vec2 d = uv - vec2(0.5);
    float r = length(d);
    vec2 off = d * r * uChroma + vec2(uSplit, 0.0);

    vec3 col;
    col.r = texture(uScene, clamp(uv + off, 0.0, 1.0)).r;
    col.g = texture(uScene, clamp(uv,       0.0, 1.0)).g;
    col.b = texture(uScene, clamp(uv - off, 0.0, 1.0)).b;

    // look VHS
    col -= sin(uv.y * 800.0) * uScan;
    col += (rnd(uv * uRes + uTime) - 0.5) * uNoise;
    col *= uTint;
    col *= 1.0 - uVignette * smoothstep(0.35, 0.9, r * 1.4142);

    fragColor = vec4(clamp(col, 0.0, 1.0), 1.0);
}
)GLSL";

// ------------------------------------------------------------------ estado GL
namespace {

struct Gpu {
    EGLContext ctx = EGL_NO_CONTEXT;
    GLuint prog = 0, vao = 0, tex = 0;
    GLint locScene = -1, locTime = -1, locRes = -1, locChroma = -1, locSplit = -1,
          locScan = -1, locNoise = -1, locJitter = -1, locVig = -1, locTint = -1;
    int w = 0, h = 0;
    bool ready = false;
};

Gpu gGpu;
bool gDisabled = false;
EGLBoolean (*gRealSwap)(EGLDisplay, EGLSurface) = nullptr;

GLuint compile(GLenum type, const char* src) {
    GLuint s = glCreateShader(type);
    glShaderSource(s, 1, &src, nullptr);
    glCompileShader(s);
    GLint ok = 0;
    glGetShaderiv(s, GL_COMPILE_STATUS, &ok);
    if (!ok) {
        char log[512] = {0};
        glGetShaderInfoLog(s, sizeof(log) - 1, nullptr, log);
        LOGE("shader no compila: %s", log);
        glDeleteShader(s);
        return 0;
    }
    return s;
}

bool initGpu(EGLContext ctx) {
    GLuint vs = compile(GL_VERTEX_SHADER, kVertSrc);
    GLuint fs = compile(GL_FRAGMENT_SHADER, kFragSrc);
    if (!vs || !fs) return false;

    GLuint p = glCreateProgram();
    glAttachShader(p, vs);
    glAttachShader(p, fs);
    glLinkProgram(p);
    glDeleteShader(vs);
    glDeleteShader(fs);
    GLint ok = 0;
    glGetProgramiv(p, GL_LINK_STATUS, &ok);
    if (!ok) {
        char log[512] = {0};
        glGetProgramInfoLog(p, sizeof(log) - 1, nullptr, log);
        LOGE("programa no enlaza: %s", log);
        return false;
    }

    gGpu = Gpu{};
    gGpu.ctx = ctx;
    gGpu.prog = p;
    gGpu.locScene  = glGetUniformLocation(p, "uScene");
    gGpu.locTime   = glGetUniformLocation(p, "uTime");
    gGpu.locRes    = glGetUniformLocation(p, "uRes");
    gGpu.locChroma = glGetUniformLocation(p, "uChroma");
    gGpu.locSplit  = glGetUniformLocation(p, "uSplit");
    gGpu.locScan   = glGetUniformLocation(p, "uScan");
    gGpu.locNoise  = glGetUniformLocation(p, "uNoise");
    gGpu.locJitter = glGetUniformLocation(p, "uJitter");
    gGpu.locVig    = glGetUniformLocation(p, "uVignette");
    gGpu.locTint   = glGetUniformLocation(p, "uTint");

    glGenVertexArrays(1, &gGpu.vao);
    glGenTextures(1, &gGpu.tex);
    gGpu.ready = true;
    LOGI("efecto inicializado");
    return true;
}

struct SavedState {
    GLint program, vao, activeTex, tex0, sampler0, drawFbo, readFbo, viewport[4];
    GLboolean colorMask[4];
    GLboolean blend, depth, cull, scissor, stencil, dither, rasterDiscard, a2c;
};

void saveState(SavedState& s) {
    glGetIntegerv(GL_CURRENT_PROGRAM, &s.program);
    glGetIntegerv(GL_VERTEX_ARRAY_BINDING, &s.vao);
    glGetIntegerv(GL_ACTIVE_TEXTURE, &s.activeTex);
    glActiveTexture(GL_TEXTURE0);
    glGetIntegerv(GL_TEXTURE_BINDING_2D, &s.tex0);
    glGetIntegerv(GL_SAMPLER_BINDING, &s.sampler0);
    glGetIntegerv(GL_DRAW_FRAMEBUFFER_BINDING, &s.drawFbo);
    glGetIntegerv(GL_READ_FRAMEBUFFER_BINDING, &s.readFbo);
    glGetIntegerv(GL_VIEWPORT, s.viewport);
    glGetBooleanv(GL_COLOR_WRITEMASK, s.colorMask);
    s.blend = glIsEnabled(GL_BLEND);
    s.depth = glIsEnabled(GL_DEPTH_TEST);
    s.cull = glIsEnabled(GL_CULL_FACE);
    s.scissor = glIsEnabled(GL_SCISSOR_TEST);
    s.stencil = glIsEnabled(GL_STENCIL_TEST);
    s.dither = glIsEnabled(GL_DITHER);
    s.rasterDiscard = glIsEnabled(GL_RASTERIZER_DISCARD);
    s.a2c = glIsEnabled(GL_SAMPLE_ALPHA_TO_COVERAGE);
}

inline void setEnabled(GLenum cap, GLboolean on) {
    if (on) glEnable(cap); else glDisable(cap);
}

void restoreState(const SavedState& s) {
    setEnabled(GL_BLEND, s.blend);
    setEnabled(GL_DEPTH_TEST, s.depth);
    setEnabled(GL_CULL_FACE, s.cull);
    setEnabled(GL_SCISSOR_TEST, s.scissor);
    setEnabled(GL_STENCIL_TEST, s.stencil);
    setEnabled(GL_DITHER, s.dither);
    setEnabled(GL_RASTERIZER_DISCARD, s.rasterDiscard);
    setEnabled(GL_SAMPLE_ALPHA_TO_COVERAGE, s.a2c);
    glColorMask(s.colorMask[0], s.colorMask[1], s.colorMask[2], s.colorMask[3]);
    glViewport(s.viewport[0], s.viewport[1], s.viewport[2], s.viewport[3]);
    glBindFramebuffer(GL_DRAW_FRAMEBUFFER, (GLuint)s.drawFbo);
    glBindFramebuffer(GL_READ_FRAMEBUFFER, (GLuint)s.readFbo);
    glActiveTexture(GL_TEXTURE0);
    glBindTexture(GL_TEXTURE_2D, (GLuint)s.tex0);
    glBindSampler(0, (GLuint)s.sampler0);
    glActiveTexture((GLenum)s.activeTex);
    glBindVertexArray((GLuint)s.vao);
    glUseProgram((GLuint)s.program);
}

float nowSeconds() {
    timespec ts{};
    clock_gettime(CLOCK_MONOTONIC, &ts);
    double t = (double)ts.tv_sec + (double)ts.tv_nsec * 1e-9;
    return (float)std::fmod(t, 1000.0);
}

void applyEffect(EGLDisplay dpy, EGLSurface surf) {
    EGLContext ctx = eglGetCurrentContext();
    if (ctx == EGL_NO_CONTEXT) return;
    if (eglGetCurrentSurface(EGL_DRAW) != surf) return;

    EGLint w = 0, h = 0;
    if (!eglQuerySurface(dpy, surf, EGL_WIDTH, &w) || !eglQuerySurface(dpy, surf, EGL_HEIGHT, &h)) return;
    if (w <= 0 || h <= 0) return;

    if (!gGpu.ready || gGpu.ctx != ctx) {
        if (!initGpu(ctx)) {
            gDisabled = true;
            LOGE("efecto desactivado (necesita OpenGL ES 3.0)");
            return;
        }
    }

    SavedState st{};
    saveState(st);

    glBindFramebuffer(GL_FRAMEBUFFER, 0);
    glActiveTexture(GL_TEXTURE0);
    glBindSampler(0, 0);
    glBindTexture(GL_TEXTURE_2D, gGpu.tex);
    if (gGpu.w != w || gGpu.h != h) {
        glTexImage2D(GL_TEXTURE_2D, 0, GL_RGB, w, h, 0, GL_RGB, GL_UNSIGNED_BYTE, nullptr);
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE);
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE);
        gGpu.w = w;
        gGpu.h = h;
    }
    glCopyTexSubImage2D(GL_TEXTURE_2D, 0, 0, 0, 0, 0, w, h);

    glDisable(GL_BLEND);
    glDisable(GL_DEPTH_TEST);
    glDisable(GL_CULL_FACE);
    glDisable(GL_SCISSOR_TEST);
    glDisable(GL_STENCIL_TEST);
    glDisable(GL_RASTERIZER_DISCARD);
    glDisable(GL_SAMPLE_ALPHA_TO_COVERAGE);
    glColorMask(GL_TRUE, GL_TRUE, GL_TRUE, GL_TRUE);
    glViewport(0, 0, w, h);

    glUseProgram(gGpu.prog);
    glUniform1i(gGpu.locScene, 0);
    glUniform1f(gGpu.locTime, nowSeconds());
    glUniform2f(gGpu.locRes, (float)w, (float)h);
    glUniform1f(gGpu.locChroma, cfg::kChroma);
    glUniform1f(gGpu.locSplit, cfg::kSplitX);
    glUniform1f(gGpu.locScan, cfg::kScanline);
    glUniform1f(gGpu.locNoise, cfg::kNoise);
    glUniform1f(gGpu.locJitter, cfg::kJitter);
    glUniform1f(gGpu.locVig, cfg::kVignette);
    glUniform3f(gGpu.locTint, cfg::kTintR, cfg::kTintG, cfg::kTintB);

    glBindVertexArray(gGpu.vao);
    glDrawArrays(GL_TRIANGLES, 0, 3);

    restoreState(st);
}

EGLBoolean hookedSwap(EGLDisplay dpy, EGLSurface surf) {
    if (!gDisabled) applyEffect(dpy, surf);
    return gRealSwap(dpy, surf);
}

// ------------------------------------------------------------ instalar el hook
struct ScanCtx {
    uintptr_t target;
    uintptr_t replacement;
    bool libFound;
    int patched;
};

int scanCallback(dl_phdr_info* info, size_t, void* data) {
    auto* c = static_cast<ScanCtx*>(data);
    if (!info->dlpi_name || !std::strstr(info->dlpi_name, "libminecraftpe.so")) return 0;
    c->libFound = true;

    const uintptr_t page = (uintptr_t)sysconf(_SC_PAGESIZE);
    for (int i = 0; i < info->dlpi_phnum; ++i) {
        const ElfW(Phdr)& ph = info->dlpi_phdr[i];
        if (ph.p_type != PT_LOAD) continue;
        if (!(ph.p_flags & PF_R) || (ph.p_flags & PF_X)) continue;  // solo datos

        uintptr_t start = (info->dlpi_addr + ph.p_vaddr + 7) & ~(uintptr_t)7;
        uintptr_t end = info->dlpi_addr + ph.p_vaddr + ph.p_memsz;
        for (uintptr_t a = start; a + sizeof(uintptr_t) <= end; a += sizeof(uintptr_t)) {
            if (*reinterpret_cast<uintptr_t*>(a) != c->target) continue;
            uintptr_t pg = a & ~(page - 1);
            if (mprotect((void*)pg, page, PROT_READ | PROT_WRITE) != 0) {
                LOGE("mprotect fallo en %p", (void*)a);
                continue;
            }
            __atomic_store_n(reinterpret_cast<uintptr_t*>(a), c->replacement, __ATOMIC_RELEASE);
            ++c->patched;
        }
    }
    return 1;  // ya encontramos la libreria
}

void* installThread(void*) {
    void* egl = dlopen("libEGL.so", RTLD_NOW);
    void* real = egl ? dlsym(egl, "eglSwapBuffers") : nullptr;
    if (!real) {
        LOGE("no se encontro eglSwapBuffers");
        return nullptr;
    }
    gRealSwap = reinterpret_cast<EGLBoolean (*)(EGLDisplay, EGLSurface)>(real);

    bool loggedWait = false;
    for (int attempt = 0; attempt < 1500; ++attempt) {  // ~5 min
        usleep(200 * 1000);
        ScanCtx c{(uintptr_t)real, (uintptr_t)&hookedSwap, false, 0};
        dl_iterate_phdr(scanCallback, &c);
        if (!c.libFound) {
            if (!loggedWait) { LOGI("esperando libminecraftpe.so..."); loggedWait = true; }
            continue;
        }
        if (c.patched > 0) {
            LOGI("hook instalado (%d entradas parcheadas)", c.patched);
            return nullptr;
        }
        if (attempt % 25 == 0) LOGI("libminecraftpe.so cargada, sin entradas de eglSwapBuffers todavia");
    }
    LOGE("no se pudo instalar el hook");
    return nullptr;
}

}  // namespace

__attribute__((constructor)) static void onLoad() {
    LOGI("RgbSplitVhs cargado");
    pthread_t t;
    if (pthread_create(&t, nullptr, installThread, nullptr) == 0) pthread_detach(t);
}
