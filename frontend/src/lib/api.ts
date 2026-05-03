const BASE = '/api/v1'

export interface PropertyInput {
  kamar_tidur:   number
  kamar_mandi:   number
  garasi:        number
  luas_tanah:    number
  luas_bangunan: number
  lokasi:        string
}

// /predict_price
export interface PredictResult {
  harga_estimasi:        number
  harga_estimasi_format: string
  model_digunakan:       string
  mape_persen:           number
  batas_segmen_idr:      number
  latency_ms:            number
}

// /classify_segment
export interface ClassifyResult {
  kelas_id:       number
  kelas_label:    string
  probabilitas:   Record<string, number>
  harga_digunakan: number
  harga_sumber:   string
  akurasi_model:  number
  latency_ms:     number
}

// /cluster_property
export interface ClusterSummary {
  cluster_id:       number
  label:            string
  jumlah_data:      number
  harga_min:        number
  harga_max:        number
  harga_median:     number
  luas_tanah_median: number
  luas_bangunan_median: number
  lokasi_dominan:   string[]
}

export interface ClusterResult {
  cluster_id:      number
  cluster_label:   string   // "Budget"
  cluster_summary: ClusterSummary
  harga_digunakan: number
  harga_sumber:    string
  silhouette_score: number
  latency_ms:      number
}

// /comparable_properties
export interface ComparableInput extends PropertyInput {
  harga:  number
  top_k?: number
}

export interface ComparableProperty {
  lokasi:        string
  kamar_tidur:   number
  kamar_mandi:   number
  garasi:        number
  luas_tanah:    number
  luas_bangunan: number
  harga:         number
  similarity?:   number
  [key: string]: unknown
}

// /chat — dengan history
export interface ChatMessage {
  role:    'user' | 'assistant'
  content: string
}

export interface ChatResult {
  reply:      string
  tools_used: string[]
}

// /feedback
export interface FeedbackInput extends PropertyInput {
  harga_prediksi: number
  harga_asli:     number
  sumber?:        string
}

export interface FeedbackResult {
  success:        boolean
  message:        string
  selisih_persen: number
}

async function post<T>(path: string, body: unknown): Promise<T> {
  const res = await fetch(`${BASE}${path}`, {
    method:  'POST',
    headers: { 'Content-Type': 'application/json' },
    body:    JSON.stringify(body),
  })
  if (!res.ok) {
    const text = await res.text()
    let msg = `HTTP ${res.status}`
    try { msg = JSON.parse(text)?.detail ?? msg } catch { msg = text || msg }
    throw new Error(msg)
  }
  return res.json()
}

export const api = {
  predictPrice:    (data: PropertyInput) =>
    post<PredictResult>('/predict_price', data),

  classifySegment: (data: PropertyInput & { harga?: number }) =>
    post<ClassifyResult>('/classify_segment', data),

  clusterProperty: (data: PropertyInput & { harga?: number }) =>
    post<ClusterResult>('/cluster_property', data),

  comparable: (data: ComparableInput) =>
    post<{ comparables: ComparableProperty[]; count: number }>('/comparable_properties', data),

  // history: array pesan sebelumnya untuk context
  chat: (message: string, history: ChatMessage[] = []) =>
    post<ChatResult>('/chat', { message, history }),

  feedback: (data: FeedbackInput) =>
    post<FeedbackResult>('/feedback', data),
}

export function formatRupiah(n: number): string {
  if (n >= 1_000_000_000) return `Rp ${(n / 1_000_000_000).toFixed(2)} M`
  if (n >= 1_000_000)     return `Rp ${(n / 1_000_000).toFixed(0)} jt`
  return `Rp ${n.toLocaleString('id-ID')}`
}

export function segmenColor(s: string): 'green' | 'amber' | 'blue' | 'red' {
  const sl = s?.toLowerCase() ?? ''
  if (sl.includes('mewah') || sl.includes('atas'))  return 'green'
  if (sl.includes('menengah'))                       return 'amber'
  if (sl.includes('murah') || sl.includes('bawah')) return 'red'
  return 'blue'
}