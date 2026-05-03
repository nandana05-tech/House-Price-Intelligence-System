'use client'

import { PageHeader, Card, SectionTitle } from '@/components/ui'

const AREA_DATA = [
  {
    nama: 'Cinere',
    avg_per_m2: 6_300_000,
    total_data: 142,
    trend: '+2.4%',
    segmen_dom: 'Menengah Atas',
    catatan: 'Akses tol Cinere-Depok, kawasan established dengan supply terbatas.',
  },
  {
    nama: 'Beji',
    avg_per_m2: 5_100_000,
    total_data: 98,
    trend: '+1.8%',
    segmen_dom: 'Menengah',
    catatan: 'Dekat Universitas Indonesia, demand tinggi dari segmen akademik.',
  },
  {
    nama: 'Sawangan',
    avg_per_m2: 4_200_000,
    total_data: 76,
    trend: '+3.1%',
    segmen_dom: 'Menengah Bawah',
    catatan: 'Area berkembang dengan potensi capital gain yang baik jangka panjang.',
  },
]

function fmt(n: number) {
  return `Rp ${(n / 1_000_000).toFixed(1)} jt/m²`
}

export default function AnalitikPage() {
  return (
    <div>
      <PageHeader
        title="Analitik Area"
        subtitle="Ringkasan kondisi pasar per lokasi berdasarkan data training"
      />
      <div className="px-8 py-6 flex flex-col gap-5">
        {AREA_DATA.map((a) => (
          <Card key={a.nama}>
            <div className="flex items-start justify-between mb-4">
              <div>
                <h2 className="text-[16px] font-semibold text-stone-900">{a.nama}</h2>
                <p className="text-[12px] text-stone-400 mt-0.5">{a.catatan}</p>
              </div>
              <span className="text-[12px] font-medium text-emerald-600 bg-emerald-50 border border-emerald-200 rounded-full px-2.5 py-0.5">
                {a.trend} / kuartal
              </span>
            </div>
            <div className="grid grid-cols-3 gap-3">
              <div className="bg-stone-50 rounded-lg p-3">
                <p className="text-[10px] text-stone-400 font-semibold uppercase tracking-wider mb-1">Avg harga/m²</p>
                <p className="text-[15px] font-semibold text-stone-900">{fmt(a.avg_per_m2)}</p>
              </div>
              <div className="bg-stone-50 rounded-lg p-3">
                <p className="text-[10px] text-stone-400 font-semibold uppercase tracking-wider mb-1">Data properti</p>
                <p className="text-[15px] font-semibold text-stone-900">{a.total_data} unit</p>
              </div>
              <div className="bg-stone-50 rounded-lg p-3">
                <p className="text-[10px] text-stone-400 font-semibold uppercase tracking-wider mb-1">Segmen dominan</p>
                <p className="text-[15px] font-semibold text-stone-900">{a.segmen_dom}</p>
              </div>
            </div>
          </Card>
        ))}
      </div>
    </div>
  )
}
