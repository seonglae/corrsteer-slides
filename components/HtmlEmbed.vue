<script setup lang="ts">
import { ref, onMounted, onBeforeUnmount } from 'vue'

// Slidev port of the article's HtmlEmbed.astro.
// Fetches a standalone embed fragment from /embeds/<src>.html, injects it,
// sets data-datafiles / data-config on the mount (embeds read these from their
// parent), then re-executes <script> tags via real script nodes so that
// document.currentScript and CDN <script src> bootstrapping both work.
const props = defineProps<{
  src: string
  title?: string
  desc?: string
  data?: string | string[]
  config?: string | Record<string, unknown>
  frameless?: boolean
}>()

const mount = ref<HTMLDivElement>()
let themeObserver: MutationObserver | null = null

function dataAttr(): string | null {
  if (Array.isArray(props.data)) return JSON.stringify(props.data)
  if (typeof props.data === 'string') return props.data
  return null
}

function configAttr(): string | null {
  if (props.config == null) return null
  return typeof props.config === 'string' ? props.config : JSON.stringify(props.config)
}

// Embeds read document.documentElement[data-theme]; Slidev toggles a `dark` class.
function syncTheme() {
  const dark = document.documentElement.classList.contains('dark')
  const next = dark ? 'dark' : 'light'
  // only write when it actually changes, otherwise embeds that observe
  // data-theme (e.g. ser-comparison) re-render on every Slidev class toggle and flicker
  if (document.documentElement.getAttribute('data-theme') !== next) {
    document.documentElement.setAttribute('data-theme', next)
  }
}

function reexecuteScripts(root: HTMLElement) {
  const scripts = Array.from(root.querySelectorAll('script'))
  scripts.forEach((old) => {
    const type = old.getAttribute('type')
    if (type && type !== 'text/javascript' && type !== 'module' && type !== '') return
    const s = document.createElement('script')
    Array.from(old.attributes).forEach((a) => s.setAttribute(a.name, a.value))
    if (!old.src) s.text = old.textContent || ''
    // replaceChild keeps document.currentScript pointing at `s` during inline run
    old.parentNode?.replaceChild(s, old)
  })
}

async function load() {
  const el = mount.value
  if (!el) return
  let name = String(props.src).replace(/^\/*/, '').replace(/^embeds\//, '')
  if (!name.endsWith('.html')) name += '.html'

  let html = ''
  try {
    const r = await fetch('/embeds/' + name, { cache: 'no-cache' })
    if (!r.ok) throw new Error(String(r.status))
    html = await r.text()
  } catch (e) {
    el.innerHTML = `<div style="color:#dc2626;font-weight:600">Embed not found: ${name}</div>`
    return
  }

  const d = dataAttr()
  const c = configAttr()
  if (d != null) el.setAttribute('data-datafiles', d)
  if (c != null) el.setAttribute('data-config', c)

  el.innerHTML = html
  // wait a tick so injected DOM is committed before scripts query it
  await Promise.resolve()
  reexecuteScripts(el)
}

onMounted(() => {
  syncTheme()
  themeObserver = new MutationObserver(syncTheme)
  themeObserver.observe(document.documentElement, { attributes: true, attributeFilter: ['class'] })
  load()
})

onBeforeUnmount(() => {
  themeObserver?.disconnect()
  themeObserver = null
})
</script>

<template>
  <figure class="html-embed">
    <figcaption v-if="title" class="html-embed__title">{{ title }}</figcaption>
    <div class="html-embed__card" :class="{ 'is-frameless': frameless }">
      <div ref="mount" :data-datafiles="dataAttr() || undefined" />
    </div>
    <figcaption v-if="desc" class="html-embed__desc" v-html="desc" />
  </figure>
</template>

<style>
/* Article palette (from app/src/styles/_variables.css) so embeds theme in both
   modes. Defined globally (non-scoped) by this component, deck-wide. */
:root {
  --neutral-600: rgb(107, 114, 128);
  --neutral-400: rgb(185, 185, 185);
  --neutral-300: rgb(228, 228, 228);
  --neutral-200: rgb(245, 245, 245);
  --primary-color: #6B5CE7;
  --primary-color-hover: #5848d0;
  --page-bg: #ffffff;
  --surface-bg: #f9f9f9;
  --text-color: rgba(0, 0, 0, 0.85);
  --muted-color: rgba(0, 0, 0, 0.6);
  --border-color: rgba(0, 0, 0, 0.1);
  --axis-color: var(--muted-color);
  --tick-color: var(--text-color);
}
html.dark, [data-theme="dark"] {
  --neutral-600: rgb(156, 163, 175);
  --neutral-400: rgb(90, 90, 90);
  --neutral-300: rgb(60, 60, 60);
  --neutral-200: rgb(40, 40, 40);
  --primary-color: #8b7cf0;
  --page-bg: #0f1115;
  --surface-bg: #16181d;
  --text-color: rgba(255, 255, 255, 0.9);
  --muted-color: rgba(255, 255, 255, 0.7);
  --border-color: rgba(255, 255, 255, 0.15);
  --axis-color: var(--muted-color);
  --tick-color: var(--muted-color);
}

.html-embed { margin: 0; position: relative; }
.html-embed__title { font-weight: 600; font-size: 0.8rem; margin: 0 0 4px; color: var(--text-color); opacity: 0.85; }
.html-embed__card {
  background: var(--surface-bg);
  color: var(--text-color);
  border: 1px solid var(--border-color);
  border-radius: 10px;
  padding: 14px;
  overflow: auto;
}
.html-embed__card.is-frameless { background: transparent; border: none; padding: 0; color: var(--text-color); }
.html-embed__desc { font-size: 0.72rem; color: var(--muted-color); margin-top: 4px; }
.plotly-graph-div { width: 100%; }
/* Force embed text/labels to follow the theme even where the embed left fill unset */
.html-embed__card,
.html-embed__card * { color: var(--text-color); }
.html-embed__card svg text { fill: var(--text-color); }
.html-embed__card select,
.html-embed__card button {
  color: var(--text-color);
  background: var(--neutral-200);
  border: 1px solid var(--border-color);
}
</style>
