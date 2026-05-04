'use client'

import { useEffect, useState } from 'react'
import { formatRupiah } from '@/lib/api'
import { PageHeader, Pill } from '@/components/ui'
import { type Lang, getLocale, getStoredLang, t } from '@/lib/i18n'

interface HistoryEntry {
  id:        string
  timestamp: number
  form:      { kamar_tidur: number; kamar_mandi: number; garasi: number; luas_tanah: number; luas_bangunan: number; lokasi: string }
  harga:     number
  segmen:    string
  cluster:   string
}

const SEGMEN_COLORS: Record<string, string> = {
  Mewah:    'green',
  Atas:     'amber',
  Menengah: 'blue',
  Murah:    'red',
}

function relativeTime(ts: number, lang: Lang): string {
  const diff = Date.now() - ts
  const mins = Math.floor(diff / 60000)
  const hrs  = Math.floor(diff / 3600000)
  if (mins < 1)  return lang === 'zh' ? '刚刚' : lang === 'en' ? 'just now' : 'baru saja'
  if (mins < 60) return lang === 'zh' ? `${mins} 分钟前` : lang === 'en' ? `${mins}m ago` : `${mins} mnt lalu`
  if (hrs  < 24) return lang === 'zh' ? `${hrs} 小时前` : lang === 'en' ? `${hrs}h ago`  : `${hrs} jam lalu`
  return new Date(ts).toLocaleDateString(getLocale(lang), { day: 'numeric', month: 'short' })
}

export default function RiwayatPage() {
  const [entries, setEntries] = useState<HistoryEntry[]>([])
  const [lang, setLang]       = useState<Lang>('id')

  useEffect(() => {
    try {
      const raw = localStorage.getItem('propvalai_history')
      if (raw) setEntries(JSON.parse(raw))
    } catch { /* ignore */ }

    setLang(getStoredLang())

    function onStorage(e: StorageEvent) {
      if (e.key === 'propvalai_lang') setLang(getStoredLang())
      if (e.key === 'propvalai_history') {
        try {
          const raw = localStorage.getItem('propvalai_history')
          if (raw) setEntries(JSON.parse(raw))
          else setEntries([])
        } catch { /* ignore */ }
      }
    }
    window.addEventListener('storage', onStorage)
    return () => window.removeEventListener('storage', onStorage)
  }, [])

  function remove(id: string) {
    const next = entries.filter((e) => e.id !== id)
    setEntries(next)
    localStorage.setItem('propvalai_history', JSON.stringify(next))
  }

  function clearAll() {
    setEntries([])
    localStorage.removeItem('propvalai_history')
  }

  const reversed = entries.slice().reverse()
  const totalEstimasi = entries.reduce((s, e) => s + e.harga, 0)

  return (
    <div className="flex flex-col h-full overflow-hidden">
      <PageHeader
        title={t(lang, 'history.pageTitle')}
        subtitle={t(lang, 'history.pageSubtitle')}
      />

      <div className="flex-1 overflow-y-auto px-8 py-6">

        {entries.length === 0 ? (
          /* ── Empty state ── */
          <div className="flex flex-col items-center justify-center h-full py-24 text-stone-400 select-none">
            <div className="relative mb-6">
              <div className="w-20 h-20 rounded-2xl bg-stone-100 flex items-center justify-center">
                <svg width="36" height="36" viewBox="0 0 36 36" fill="none">
                  <circle cx="18" cy="18" r="14" stroke="currentColor" strokeWidth="1.5" strokeDasharray="4 2"/>
                  <path d="M18 11v7l4 3" stroke="currentColor" strokeWidth="1.5" strokeLinecap="round" strokeLinejoin="round"/>
                </svg>
              </div>
              <div className="absolute -top-1 -right-1 w-5 h-5 rounded-full bg-amber-100 flex items-center justify-center">
                <span className="text-[9px] font-bold text-amber-600">0</span>
              </div>
            </div>
            <p className="text-[15px] font-semibold text-stone-700 mb-1">{t(lang, 'history.emptyTitle')}</p>
            <p className="text-[13px] text-stone-400">{t(lang, 'history.emptySubtitle')}</p>
            <div className="mt-6 flex items-center gap-2 text-[12px] text-stone-300">
              <svg width="14" height="14" viewBox="0 0 14 14" fill="none">
                <path d="M7 1v6M4 4l3-3 3 3" stroke="currentColor" strokeWidth="1.3" strokeLinecap="round" strokeLinejoin="round"/>
                <path d="M2 10h10" stroke="currentColor" strokeWidth="1.3" strokeLinecap="round"/>
                <path d="M2 13h10" stroke="currentColor" strokeWidth="1.3" strokeLinecap="round"/>
              </svg>
              Prediksi akan tersimpan otomatis
            </div>
          </div>
        ) : (
          <>
            {/* ── Summary strip ── */}
            <div className="grid grid-cols-3 gap-4 mb-6 animate-slide-up">
              <div className="bg-white border border-stone-200 rounded-xl p-4 flex flex-col gap-1">
                <span className="text-[10px] font-semibold uppercase tracking-widest text-stone-400">Total Prediksi</span>
                <span className="text-2xl font-semibold text-stone-900">{entries.length}</span>
                <span className="text-[11px] text-stone-400">sesi ini</span>
              </div>
              <div className="bg-white border border-stone-200 rounded-xl p-4 flex flex-col gap-1">
                <span className="text-[10px] font-semibold uppercase tracking-widest text-stone-400">Estimasi Tertinggi</span>
                <span className="text-xl font-semibold text-amber-600 truncate">
                  {formatRupiah(Math.max(...entries.map(e => e.harga)))}
                </span>
                <span className="text-[11px] text-stone-400">{entries.find(e => e.harga === Math.max(...entries.map(x => x.harga)))?.form.lokasi}</span>
              </div>
              <div className="bg-white border border-stone-200 rounded-xl p-4 flex flex-col gap-1">
                <span className="text-[10px] font-semibold uppercase tracking-widest text-stone-400">Rata-rata Estimasi</span>
                <span className="text-xl font-semibold text-stone-900 truncate">
                  {formatRupiah(Math.round(totalEstimasi / entries.length))}
                </span>
                <span className="text-[11px] text-stone-400">per properti</span>
              </div>
            </div>

            {/* ── List header ── */}
            <div className="flex items-center justify-between mb-3">
              <p className="text-[11px] font-semibold uppercase tracking-widest text-stone-400">
                {reversed.length} {t(lang, 'history.pageTitle').toLowerCase()}
              </p>
              <button
                onClick={clearAll}
                className="text-[11px] text-stone-300 hover:text-red-400 transition-colors"
              >
                Hapus semua
              </button>
            </div>

            {/* ── Cards ── */}
            <div className="flex flex-col gap-2.5 animate-slide-up stagger">
              {reversed.map((e, idx) => {
                const segColor = (SEGMEN_COLORS[e.segmen] ?? 'stone') as 'green' | 'amber' | 'blue' | 'red' | 'stone'
                const rank = reversed.length - idx
                return (
                  <div
                    key={e.id}
                    className="group bg-white border border-stone-200 rounded-xl px-5 py-4 hover:border-stone-300 hover:shadow-sm transition-all duration-200 flex items-center gap-4"
                  >
                    {/* Rank badge */}
                    <div className="flex-shrink-0 w-8 h-8 rounded-lg bg-stone-50 border border-stone-100 flex items-center justify-center">
                      <span className="text-[11px] font-bold text-stone-400">#{rank}</span>
                    </div>

                    {/* Main info */}
                    <div className="flex-1 min-w-0">
                      <div className="flex items-center gap-2 mb-1.5 flex-wrap">
                        <span className="text-[14px] font-semibold text-stone-900 leading-none">{e.form.lokasi}</span>
                        <Pill color={segColor}>{e.segmen}</Pill>
                        <Pill color="stone">{e.cluster}</Pill>
                      </div>
                      <div className="flex items-center gap-3 text-[11px] text-stone-400">
                        <span className="flex items-center gap-1">
                          <svg width="10" height="10" viewBox="0 0 10 10" fill="none">
                            <rect x="1" y="1" width="8" height="8" rx="1.5" stroke="currentColor" strokeWidth="1.2"/>
                            <path d="M3 5h4M3 7h2" stroke="currentColor" strokeWidth="1.2" strokeLinecap="round"/>
                          </svg>
                          {e.form.kamar_tidur}KT · {e.form.kamar_mandi}KM · {e.form.garasi}G
                        </span>
                        <span className="text-stone-200">|</span>
                        <span>{e.form.luas_tanah}m² · {e.form.luas_bangunan}m²</span>
                        <span className="text-stone-200">|</span>
                        <span>{relativeTime(e.timestamp, lang)}</span>
                      </div>
                    </div>

                    {/* Price + delete */}
                    <div className="flex-shrink-0 text-right flex flex-col items-end gap-1.5">
                      <span className="text-[16px] font-semibold text-amber-600">{formatRupiah(e.harga)}</span>
                      <button
                        onClick={() => remove(e.id)}
                        className="opacity-0 group-hover:opacity-100 text-[10px] text-stone-300 hover:text-red-400 transition-all"
                      >
                        {t(lang, 'history.delete')}
                      </button>
                    </div>
                  </div>
                )
              })}
            </div>
          </>
        )}
      </div>
    </div>
  )
}
