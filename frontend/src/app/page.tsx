'use client'

import { useState } from 'react'
import {
  api, formatRupiah, segmenColor,
  type PropertyInput, type PredictResult,
  type ClassifyResult, type ClusterResult, type ComparableProperty,
} from '@/lib/api'
import { Card, SectionTitle, Input, Select, Button, Metric, Skeleton, PageHeader, Pill } from '@/components/ui'
import ChatPanel from '@/components/ChatPanel'

const LOKASI_OPTIONS = [
  'Babadan', 'Beji', 'Bojong Sari', 'Ciangsana', 'Cibubur',
  'Cilangkap', 'Cilodong', 'Cimanggis', 'Cinangka', 'Cinere',
  'Cipayung', 'Cisalak', 'Citayam', 'Depok I', 'Depok II',
  'Gandul', 'Grand Depok City', 'Harjamukti', 'Kalimanggis',
  'Kelapa Dua', 'Krukut', 'Kukusan', 'Limo', 'Margonda',
  'Mekarsari', 'Pancoran Mas', 'Pangkalan Jati', 'Pasir Putih',
  'Rangkapanjaya', 'Sawangan', 'Studio Alam', 'Sukatani',
  'Sukmajaya', 'Tanah Baru', 'Tapos', 'Tirtajaya', 'Tugu',
]

interface AnalysisResult {
  predict:    PredictResult
  classify:   ClassifyResult
  cluster:    ClusterResult
  comparable: ComparableProperty[]
}

export default function PrediksiPage() {
  const [form, setForm] = useState<PropertyInput>({
    kamar_tidur:   3,
    kamar_mandi:   2,
    garasi:        1,
    luas_tanah:    120,
    luas_bangunan: 90,
    lokasi:        'Cinere',
  })

  const [loading,         setLoading]         = useState(false)
  const [result,          setResult]          = useState<AnalysisResult | null>(null)
  const [error,           setError]           = useState<string | null>(null)
  const [hargaAsli,       setHargaAsli]       = useState('')
  const [feedbackSent,    setFeedbackSent]    = useState(false)
  const [feedbackLoading, setFeedbackLoading] = useState(false)
  const [feedbackError,   setFeedbackError]   = useState<string | null>(null)

  function setField(key: keyof PropertyInput, val: string) {
    setForm(prev => ({
      ...prev,
      [key]: key === 'lokasi' ? val : Number(val),
    }))
  }

  async function handleAnalyze() {
    setLoading(true)
    setError(null)
    setResult(null)
    setFeedbackSent(false)
    setFeedbackError(null)
    setHargaAsli('')

    try {
      // Step 1: predict, classify, cluster paralel
      const [predict, classify, cluster] = await Promise.all([
        api.predictPrice(form),
        api.classifySegment(form),
        api.clusterProperty(form),
      ])

      // Step 2: comparable pakai harga_estimasi dari predict
      const comp = await api.comparable({
        ...form,
        harga: predict.harga_estimasi,
        top_k: 3,
      }).catch(() => ({ comparables: [], count: 0 }))

      setResult({ predict, classify, cluster, comparable: comp.comparables ?? [] })
    } catch (e: unknown) {
      setError(e instanceof Error ? e.message : 'Terjadi kesalahan.')
    } finally {
      setLoading(false)
    }
  }

  async function handleFeedback() {
    if (!result || !hargaAsli) return
    const hargaAsliNum = Number(hargaAsli)
    if (hargaAsliNum < 10_000_000) {
      setFeedbackError('Harga minimal Rp 10.000.000')
      return
    }
    setFeedbackLoading(true)
    setFeedbackError(null)
    try {
      await api.feedback({
        ...form,
        harga_prediksi: result.predict.harga_estimasi,
        harga_asli:     hargaAsliNum,
        sumber:         'user_feedback',
      })
      setFeedbackSent(true)
    } catch (e: unknown) {
      setFeedbackError(e instanceof Error ? e.message : 'Gagal mengirim feedback.')
    } finally {
      setFeedbackLoading(false)
    }
  }

  const harga      = result?.predict?.harga_estimasi
  const hargaFmt   = result?.predict?.harga_estimasi_format
  const mape       = result?.predict?.mape_persen
  const model      = result?.predict?.model_digunakan

  // classify: pakai kelas_label
  const segmen     = result?.classify?.kelas_label ?? '—'
  const probTop    = result?.classify?.probabilitas
    ? Object.entries(result.classify.probabilitas).sort((a, b) => b[1] - a[1])[0]
    : null

  // cluster: pakai cluster_label + cluster_summary
  const clusterLabel   = result?.cluster?.cluster_label ?? '—'
  const clusterSummary = result?.cluster?.cluster_summary

  return (
    <div className="flex flex-col h-full">
      <PageHeader
        title="Prediksi Harga Properti"
        subtitle="Estimasi harga, segmen, dan klaster properti menggunakan model ML"
      />

      <div className="flex flex-1 overflow-hidden">
        <div className="flex-1 overflow-y-auto px-8 py-6 flex flex-col gap-5">

          {/* Form */}
          <Card>
            <SectionTitle>Parameter Properti</SectionTitle>
            <div className="grid grid-cols-3 gap-4 mb-4">
              <Input label="Kamar Tidur"        type="number" min={1}  max={20} value={form.kamar_tidur}   onChange={v => setField('kamar_tidur', v)} />
              <Input label="Kamar Mandi"        type="number" min={1}  max={10} value={form.kamar_mandi}   onChange={v => setField('kamar_mandi', v)} />
              <Input label="Garasi"             type="number" min={0}  max={10} value={form.garasi}        onChange={v => setField('garasi', v)} />
              <Input label="Luas Tanah (m²)"    type="number" min={1}           value={form.luas_tanah}    onChange={v => setField('luas_tanah', v)} />
              <Input label="Luas Bangunan (m²)" type="number" min={1}           value={form.luas_bangunan} onChange={v => setField('luas_bangunan', v)} />
              <Select label="Lokasi" value={form.lokasi} options={LOKASI_OPTIONS} onChange={v => setField('lokasi', v)} />
            </div>
            <Button onClick={handleAnalyze} loading={loading} className="w-full">
              {loading ? 'Menganalisa...' : 'Analisa Properti'}
            </Button>
            {error && (
              <p className="mt-3 text-[12px] text-red-500 bg-red-50 border border-red-100 rounded-lg px-3 py-2">{error}</p>
            )}
          </Card>

          {/* Skeleton */}
          {loading && (
            <Card>
              <div className="grid grid-cols-3 gap-4 mb-5">
                <Skeleton className="h-[76px]" /><Skeleton className="h-[76px]" /><Skeleton className="h-[76px]" />
              </div>
              <Skeleton className="h-4 w-32 mb-3" />
              <div className="flex flex-col gap-2">
                <Skeleton className="h-14" /><Skeleton className="h-14" /><Skeleton className="h-14" />
              </div>
            </Card>
          )}

          {/* Results */}
          {result && !loading && (
            <div className="animate-slide-up flex flex-col gap-5">

              {/* Metrics utama */}
              <Card>
                <SectionTitle>Hasil Analisa</SectionTitle>
                <div className="grid grid-cols-3 gap-3 mb-4">
                  <Metric
                    label="Estimasi Harga"
                    value={hargaFmt ?? formatRupiah(harga ?? 0)}
                    sub={`MAPE ±${mape?.toFixed(2)}%`}
                    accent
                  />
                  <Metric
                    label="Segmen"
                    value={segmen}
                    sub={probTop ? `Probabilitas: ${(probTop[1] * 100).toFixed(1)}%` : 'Klasifikasi model'}
                  />
                  <Metric
                    label="Klaster"
                    value={clusterLabel}
                    sub={clusterSummary ? `Median: ${formatRupiah(clusterSummary.harga_median)}` : undefined}
                  />
                </div>

                {/* Detail cluster */}
                {clusterSummary && (
                  <div className="bg-stone-50 rounded-lg p-3 mb-4 text-[11px] text-stone-500 grid grid-cols-2 gap-1.5">
                    <span>📍 Lokasi dominan: <strong className="text-stone-700">{clusterSummary.lokasi_dominan.join(', ')}</strong></span>
                    <span>🏠 Luas tanah median: <strong className="text-stone-700">{clusterSummary.luas_tanah_median}m²</strong></span>
                    <span>💰 Rentang: <strong className="text-stone-700">{formatRupiah(clusterSummary.harga_min)} – {formatRupiah(clusterSummary.harga_max)}</strong></span>
                    <span>📊 Jumlah data: <strong className="text-stone-700">{clusterSummary.jumlah_data} properti</strong></span>
                  </div>
                )}

                {/* Model info */}
                <div className="flex items-center gap-2 mb-4">
                  <span className="text-[11px] text-stone-400">Model digunakan:</span>
                  <Pill color="stone">{model ?? '—'}</Pill>
                </div>

                {/* Feedback */}
                <div className="border-t border-stone-100 pt-4">
                  <p className="text-[11px] font-semibold uppercase tracking-widest text-stone-400 mb-3">Koreksi Harga</p>
                  {feedbackSent ? (
                    <div className="flex items-center gap-2 text-[13px] text-emerald-600 bg-emerald-50 border border-emerald-200 rounded-lg px-3 py-2">
                      <svg width="14" height="14" viewBox="0 0 14 14" fill="none">
                        <circle cx="7" cy="7" r="6" stroke="currentColor" strokeWidth="1.5"/>
                        <path d="M4.5 7L6 8.5L9.5 5" stroke="currentColor" strokeWidth="1.5" strokeLinecap="round"/>
                      </svg>
                      Feedback terkirim — terima kasih!
                    </div>
                  ) : (
                    <>
                      <div className="flex gap-2">
                        <input
                          type="number"
                          placeholder="Harga aktual (min Rp 10.000.000)..."
                          value={hargaAsli}
                          onChange={e => setHargaAsli(e.target.value)}
                          className="flex-1 px-3 py-2 text-[13px] bg-stone-50 border border-stone-200 rounded-lg text-stone-900 placeholder-stone-300 focus:outline-none focus:border-amber-400 focus:ring-1 focus:ring-amber-400/30 transition-all"
                        />
                        <Button variant="secondary" onClick={handleFeedback} loading={feedbackLoading}>Kirim</Button>
                      </div>
                      {feedbackError && <p className="mt-2 text-[11px] text-red-500">{feedbackError}</p>}
                    </>
                  )}
                </div>
              </Card>

              {/* Comparable */}
              {result.comparable.length > 0 && (
                <Card>
                  <SectionTitle>Properti Pembanding</SectionTitle>
                  <div className="flex flex-col gap-2">
                    {result.comparable.map((p, i) => (
                      <div key={i} className="flex items-center justify-between px-4 py-3 bg-stone-50 rounded-lg border border-stone-100 hover:border-stone-200 transition-colors">
                        <div>
                          <div className="text-[13px] font-medium text-stone-800">
                            {p.lokasi} — {p.kamar_tidur}KT {p.kamar_mandi}KM {p.garasi}Garasi
                          </div>
                          <div className="text-[11px] text-stone-400 mt-0.5">
                            {p.luas_tanah}m² tanah · {p.luas_bangunan}m² bangunan
                          </div>
                          {p.similarity !== undefined && (
                            <div className="mt-1.5 flex items-center gap-2">
                              <div className="w-24 h-1 bg-stone-200 rounded-full overflow-hidden">
                                <div className="h-full bg-amber-400 rounded-full" style={{ width: `${Math.round(p.similarity * 100)}%` }} />
                              </div>
                              <span className="text-[10px] text-stone-400">{Math.round(p.similarity * 100)}% mirip</span>
                            </div>
                          )}
                        </div>
                        <div className="text-right ml-4 flex-shrink-0">
                          <div className="text-[14px] font-semibold text-amber-600">{formatRupiah(p.harga)}</div>
                        </div>
                      </div>
                    ))}
                  </div>
                </Card>
              )}
            </div>
          )}
        </div>

        {/* Chat panel */}
        <div className="w-[300px] flex-shrink-0 border-l border-stone-100 flex flex-col">
          <ChatPanel lokasi={form.lokasi} hargaPrediksi={harga} />
        </div>
      </div>
    </div>
  )
}