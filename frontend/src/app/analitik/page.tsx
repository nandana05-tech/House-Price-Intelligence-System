'use client'

import { useEffect, useState } from 'react'
import { PageHeader } from '@/components/ui'
import { type Lang, getStoredLang, t } from '@/lib/i18n'
import { api, type AreaAnalytics } from '@/lib/api'

function fmt(n: number) {
  return `Rp ${(n / 1_000_000).toFixed(1)} jt`
}

const SEGMENT_COLORS: Record<string, { bg: string; text: string; dot: string }> = {
  Mewah:    { bg: 'bg-violet-50',  text: 'text-violet-700',  dot: 'bg-violet-400' },
  Atas:     { bg: 'bg-amber-50',   text: 'text-amber-700',   dot: 'bg-amber-400'  },
  Menengah: { bg: 'bg-sky-50',     text: 'text-sky-700',     dot: 'bg-sky-400'    },
  Murah:    { bg: 'bg-emerald-50', text: 'text-emerald-700', dot: 'bg-emerald-400'},
}
function segmentStyle(s: string) {
  return SEGMENT_COLORS[s] ?? { bg: 'bg-stone-100', text: 'text-stone-600', dot: 'bg-stone-400' }
}

export default function AnalitikPage() {
  const [lang, setLang]     = useState<Lang>('id')
  const [data, setData]     = useState<AreaAnalytics[]>([])
  const [loading, setLoading] = useState(true)
  const [error, setError]   = useState<string | null>(null)

  useEffect(() => {
    setLang(getStoredLang())
    function onStorage(e: StorageEvent) {
      if (e.key === 'propvalai_lang') setLang(getStoredLang())
    }
    window.addEventListener('storage', onStorage)

    api.analytics()
      .then(res => setData(res))
      .catch(err => setError(err.message))
      .finally(() => setLoading(false))

    return () => window.removeEventListener('storage', onStorage)
  }, [])

  // compute max for bar chart scaling
  const maxPrice = Math.max(...data.map(d => d.avg_per_m2), 1)

  return (
    <div className="flex flex-col h-full overflow-hidden">
      <PageHeader
        title={t(lang, 'analytics.pageTitle')}
        subtitle={t(lang, 'analytics.pageSubtitle')}
      />

      <div className="flex-1 overflow-y-auto px-8 py-6">

        {/* ── Summary strip ── */}
        {!loading && !error && data.length > 0 && (
          <div className="grid grid-cols-3 gap-4 mb-6 animate-slide-up">
            <div className="bg-white border border-stone-200 rounded-xl p-4 flex flex-col gap-1">
              <span className="text-[10px] font-semibold uppercase tracking-widest text-stone-400">Total Area</span>
              <span className="text-2xl font-semibold text-stone-900">{data.length}</span>
              <span className="text-[11px] text-stone-400">{t(lang, 'analytics.pageSubtitle').split(' ').slice(0,3).join(' ')}…</span>
            </div>
            <div className="bg-white border border-stone-200 rounded-xl p-4 flex flex-col gap-1">
              <span className="text-[10px] font-semibold uppercase tracking-widest text-stone-400">{t(lang, 'analytics.avgPricePerM2')}</span>
              <span className="text-2xl font-semibold text-amber-600">
                {fmt(data.reduce((s, d) => s + d.avg_per_m2, 0) / data.length)}/m²
              </span>
              <span className="text-[11px] text-stone-400">rata-rata semua area</span>
            </div>
            <div className="bg-white border border-stone-200 rounded-xl p-4 flex flex-col gap-1">
              <span className="text-[10px] font-semibold uppercase tracking-widest text-stone-400">{t(lang, 'analytics.propertyData')}</span>
              <span className="text-2xl font-semibold text-stone-900">
                {data.reduce((s, d) => s + d.total_data, 0).toLocaleString()}
              </span>
              <span className="text-[11px] text-stone-400">{t(lang, 'analytics.unit')}</span>
            </div>
          </div>
        )}

        {/* ── Loading skeleton ── */}
        {loading && (
          <div className="flex flex-col gap-3">
            {[1,2,3,4,5].map(i => (
              <div key={i} className="bg-white border border-stone-200 rounded-xl p-5">
                <div className="flex items-center justify-between mb-3">
                  <div className="skeleton h-4 w-28 rounded" />
                  <div className="skeleton h-5 w-20 rounded-full" />
                </div>
                <div className="skeleton h-2.5 w-full rounded-full mb-3" />
                <div className="flex gap-3">
                  <div className="skeleton h-10 w-1/3 rounded-lg" />
                  <div className="skeleton h-10 w-1/3 rounded-lg" />
                  <div className="skeleton h-10 w-1/3 rounded-lg" />
                </div>
              </div>
            ))}
          </div>
        )}

        {/* ── Error ── */}
        {error && !loading && (
          <div className="flex flex-col items-center justify-center py-24 text-stone-400">
            <div className="w-12 h-12 rounded-full bg-red-50 border border-red-100 flex items-center justify-center mb-4">
              <svg width="20" height="20" viewBox="0 0 20 20" fill="none">
                <circle cx="10" cy="10" r="8" stroke="#f87171" strokeWidth="1.5"/>
                <path d="M10 6v4M10 13h.01" stroke="#f87171" strokeWidth="1.5" strokeLinecap="round"/>
              </svg>
            </div>
            <p className="text-[13px] font-medium text-red-500">{t(lang, 'analytics.loadError')}</p>
            <p className="text-[11px] text-stone-300 mt-1">{error}</p>
          </div>
        )}

        {/* ── Area cards ── */}
        {!loading && !error && (
          <div className="flex flex-col gap-3 animate-slide-up stagger">
            {data.map((a, idx) => {
              const seg  = segmentStyle(a.segmen_dom)
              const barW = Math.round((a.avg_per_m2 / maxPrice) * 100)
              const hasTrend = a.trend !== 'N/A'
              return (
                <div
                  key={a.nama}
                  className="bg-white border border-stone-200 rounded-xl p-5 hover:border-stone-300 hover:shadow-sm transition-all duration-200"
                >
                  {/* Header row */}
                  <div className="flex items-start justify-between mb-3">
                    <div className="flex items-center gap-2.5">
                      <span className="w-6 h-6 rounded-full bg-stone-100 flex items-center justify-center text-[10px] font-bold text-stone-500 flex-shrink-0">
                        {idx + 1}
                      </span>
                      <div>
                        <h2 className="text-[14px] font-semibold text-stone-900 leading-none">{a.nama}</h2>
                        <p className="text-[11px] text-stone-400 mt-0.5">{t(lang, 'analytics.catatan')}</p>
                      </div>
                    </div>
                    <div className="flex items-center gap-2 flex-shrink-0">
                      {/* Segment pill */}
                      <span className={`inline-flex items-center gap-1.5 px-2.5 py-1 rounded-full text-[11px] font-medium ${seg.bg} ${seg.text}`}>
                        <span className={`w-1.5 h-1.5 rounded-full ${seg.dot}`} />
                        {a.segmen_dom}
                      </span>
                      {/* Trend badge */}
                      <span className={`text-[11px] font-medium px-2.5 py-1 rounded-full border ${
                        hasTrend
                          ? 'text-emerald-700 bg-emerald-50 border-emerald-200'
                          : 'text-stone-400 bg-stone-50 border-stone-200'
                      }`}>
                        {hasTrend ? a.trend : t(lang, 'analytics.noTrend')} {t(lang, 'analytics.errorRate')}
                      </span>
                    </div>
                  </div>

                  {/* Price bar */}
                  <div className="mb-4">
                    <div className="flex items-center justify-between mb-1">
                      <span className="text-[10px] text-stone-400">{t(lang, 'analytics.avgPricePerM2')}</span>
                      <span className="text-[12px] font-semibold text-amber-600">{fmt(a.avg_per_m2)}/m²</span>
                    </div>
                    <div className="h-1.5 w-full bg-stone-100 rounded-full overflow-hidden">
                      <div
                        className="h-full bg-gradient-to-r from-amber-400 to-amber-500 rounded-full transition-all duration-700"
                        style={{ width: `${barW}%` }}
                      />
                    </div>
                  </div>

                  {/* Stats row */}
                  <div className="grid grid-cols-2 gap-3">
                    <div className="bg-stone-50 rounded-lg px-3 py-2.5 flex flex-col gap-0.5">
                      <span className="text-[10px] text-stone-400 font-medium uppercase tracking-wider">{t(lang, 'analytics.propertyData')}</span>
                      <span className="text-[15px] font-semibold text-stone-900">
                        {a.total_data.toLocaleString()} <span className="text-[11px] font-normal text-stone-400">{t(lang, 'analytics.unit')}</span>
                      </span>
                    </div>
                    <div className="bg-stone-50 rounded-lg px-3 py-2.5 flex flex-col gap-0.5">
                      <span className="text-[10px] text-stone-400 font-medium uppercase tracking-wider">{t(lang, 'analytics.dominantSegment')}</span>
                      <span className={`text-[15px] font-semibold ${seg.text}`}>{a.segmen_dom}</span>
                    </div>
                  </div>
                </div>
              )
            })}
          </div>
        )}

        {/* ── Empty ── */}
        {!loading && !error && data.length === 0 && (
          <div className="flex flex-col items-center justify-center py-24 text-stone-400">
            <svg className="mb-4 opacity-20" width="48" height="48" viewBox="0 0 48 48" fill="none">
              <rect x="8" y="28" width="8" height="12" rx="2" fill="currentColor"/>
              <rect x="20" y="18" width="8" height="22" rx="2" fill="currentColor"/>
              <rect x="32" y="8" width="8" height="32" rx="2" fill="currentColor"/>
            </svg>
            <p className="text-[14px]">Tidak ada data analitik</p>
          </div>
        )}
      </div>
    </div>
  )
}
