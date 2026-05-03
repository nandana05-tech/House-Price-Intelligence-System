'use client'

import { useEffect, useState } from 'react'
import { formatRupiah } from '@/lib/api'
import { PageHeader, Card, Pill } from '@/components/ui'

interface HistoryEntry {
  id:        string
  timestamp: number
  form:      { kamar_tidur: number; kamar_mandi: number; garasi: number; luas_tanah: number; luas_bangunan: number; lokasi: string }
  harga:     number
  segmen:    string
  cluster:   string
}

export default function RiwayatPage() {
  const [entries, setEntries] = useState<HistoryEntry[]>([])

  useEffect(() => {
    try {
      const raw = localStorage.getItem('propvalai_history')
      if (raw) setEntries(JSON.parse(raw))
    } catch { /* ignore */ }
  }, [])

  function remove(id: string) {
    const next = entries.filter((e) => e.id !== id)
    setEntries(next)
    localStorage.setItem('propvalai_history', JSON.stringify(next))
  }

  return (
    <div>
      <PageHeader
        title="Riwayat Prediksi"
        subtitle="Daftar prediksi yang telah dilakukan dalam sesi ini"
      />
      <div className="px-8 py-6">
        {entries.length === 0 ? (
          <div className="text-center py-20 text-stone-400">
            <svg className="mx-auto mb-4 opacity-30" width="40" height="40" viewBox="0 0 40 40" fill="none">
              <circle cx="20" cy="20" r="18" stroke="currentColor" strokeWidth="2"/>
              <path d="M20 12v8l5 4" stroke="currentColor" strokeWidth="2" strokeLinecap="round"/>
            </svg>
            <p className="text-[14px]">Belum ada riwayat prediksi</p>
            <p className="text-[12px] mt-1">Lakukan analisa properti terlebih dahulu</p>
          </div>
        ) : (
          <div className="flex flex-col gap-3">
            {entries.slice().reverse().map((e) => (
              <Card key={e.id} className="flex items-center justify-between gap-4">
                <div className="flex-1">
                  <div className="flex items-center gap-2 mb-1">
                    <span className="text-[13px] font-medium text-stone-800">{e.form.lokasi}</span>
                    <Pill color="amber">{e.segmen}</Pill>
                    <Pill color="blue">{e.cluster}</Pill>
                  </div>
                  <div className="text-[12px] text-stone-400">
                    {e.form.kamar_tidur} KT · {e.form.kamar_mandi} KM · {e.form.garasi} Garasi ·{' '}
                    {e.form.luas_tanah}m² tanah · {e.form.luas_bangunan}m² bangunan
                  </div>
                  <div className="text-[11px] text-stone-300 mt-1">
                    {new Date(e.timestamp).toLocaleString('id-ID')}
                  </div>
                </div>
                <div className="text-right flex-shrink-0">
                  <div className="text-[16px] font-semibold text-amber-600">{formatRupiah(e.harga)}</div>
                  <button
                    onClick={() => remove(e.id)}
                    className="text-[11px] text-stone-300 hover:text-red-400 mt-1 transition-colors"
                  >
                    Hapus
                  </button>
                </div>
              </Card>
            ))}
          </div>
        )}
      </div>
    </div>
  )
}
