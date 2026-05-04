export type Lang = 'id' | 'en' | 'zh'

const LANG_KEY = 'propvalai_lang'

type Dict = Record<string, string>

const messages: Record<Lang, Dict> = {
  id: {
    // nav/layout
    'app.langLabel': 'Bahasa',
    'nav.menu': 'Menu',
    'nav.prediksi': 'Prediksi',
    'nav.riwayat': 'Riwayat',
    'nav.analitik': 'Analitik',
    'app.modelActive': 'Model aktif',
    'app.estimasiProperti': 'Estimasi Properti',

    // page - prediksi
    'pred.pageTitle': 'Prediksi Harga Properti',
    'pred.pageSubtitle': 'Estimasi harga, segmen, dan klaster properti menggunakan model ML',
    'pred.parameterTitle': 'Parameter Properti',
    'pred.bedroom': 'Kamar Tidur',
    'pred.bathroom': 'Kamar Mandi',
    'pred.garage': 'Garasi',
    'pred.landArea': 'Luas Tanah (m²)',
    'pred.buildingArea': 'Luas Bangunan (m²)',
    'pred.location': 'Lokasi',
    'pred.analyzing': 'Menganalisa...',
    'pred.analyze': 'Analisa Properti',
    'pred.errorGeneric': 'Terjadi kesalahan.',
    'pred.resultTitle': 'Hasil Analisa',
    'pred.estimatedPrice': 'Estimasi Harga',
    'pred.segment': 'Segmen',
    'pred.cluster': 'Klaster',
    'pred.modelUsed': 'Model digunakan:',
    'pred.probability': 'Probabilitas',
    'pred.clusterMedian': 'Median',
    'pred.clusterDominantLocation': 'Lokasi dominan',
    'pred.clusterLandMedian': 'Luas tanah median',
    'pred.clusterRange': 'Rentang',
    'pred.clusterDataCount': 'Jumlah data',
    'pred.properties': 'properti',
    'pred.priceCorrection': 'Koreksi Harga',
    'pred.feedbackSent': 'Feedback terkirim — terima kasih!',
    'pred.actualPricePlaceholder': 'Harga aktual (min Rp 10.000.000)...',
    'pred.send': 'Kirim',
    'pred.feedbackMinPrice': 'Harga minimal Rp 10.000.000',
    'pred.feedbackError': 'Gagal mengirim feedback.',
    'pred.comparableTitle': 'Properti Pembanding',
    'pred.similarity': 'mirip',

    // chat
    'chat.title': 'Asisten AI',
    'chat.subtitle': 'Tanya soal properti',
    'chat.location': 'Lokasi',
    'chat.welcome': 'Halo! Saya siap membantu menjawab pertanyaan seputar estimasi harga dan pasar properti di Depok.',
    'chat.placeholder': 'Tulis pertanyaan...',
    'chat.error': 'Terjadi kesalahan.',
    'chat.quick1': 'Bagaimana cara harga dihitung?',
    'chat.quick2': 'Apa itu segmen menengah atas?',
    'chat.quick3': 'Faktor apa yang mempengaruhi harga?',

    // riwayat
    'history.pageTitle': 'Riwayat Prediksi',
    'history.pageSubtitle': 'Daftar prediksi yang telah dilakukan dalam sesi ini',
    'history.emptyTitle': 'Belum ada riwayat prediksi',
    'history.emptySubtitle': 'Lakukan analisa properti terlebih dahulu',
    'history.delete': 'Hapus',

    // analitik
    'analytics.pageTitle': 'Analitik Area',
    'analytics.pageSubtitle': 'Ringkasan kondisi pasar per lokasi berdasarkan data training',
    'analytics.trendPerQuarter': '/ kuartal',
    'analytics.avgPricePerM2': 'Avg harga/m²',
    'analytics.propertyData': 'Data properti',
    'analytics.unit': 'unit',
    'analytics.dominantSegment': 'Segmen dominan',
    'analytics.errorRate': 'tingkat error',
    'analytics.noTrend': 'Belum ada',
    'analytics.catatan': 'Data dianalisis secara dinamis berdasarkan model klasifikasi.',
    'analytics.loadError': 'Gagal memuat analitik',
  },
  en: {
    'app.langLabel': 'Language',
    'nav.menu': 'Menu',
    'nav.prediksi': 'Prediction',
    'nav.riwayat': 'History',
    'nav.analitik': 'Analytics',
    'app.modelActive': 'Model active',
    'app.estimasiProperti': 'Property Estimation',

    'pred.pageTitle': 'Property Price Prediction',
    'pred.pageSubtitle': 'Estimate price, segment, and cluster using ML models',
    'pred.parameterTitle': 'Property Parameters',
    'pred.bedroom': 'Bedrooms',
    'pred.bathroom': 'Bathrooms',
    'pred.garage': 'Garage',
    'pred.landArea': 'Land Area (m²)',
    'pred.buildingArea': 'Building Area (m²)',
    'pred.location': 'Location',
    'pred.analyzing': 'Analyzing...',
    'pred.analyze': 'Analyze Property',
    'pred.errorGeneric': 'An error occurred.',
    'pred.resultTitle': 'Analysis Result',
    'pred.estimatedPrice': 'Estimated Price',
    'pred.segment': 'Segment',
    'pred.cluster': 'Cluster',
    'pred.modelUsed': 'Model used:',
    'pred.probability': 'Probability',
    'pred.clusterMedian': 'Median',
    'pred.clusterDominantLocation': 'Dominant locations',
    'pred.clusterLandMedian': 'Median land area',
    'pred.clusterRange': 'Range',
    'pred.clusterDataCount': 'Data count',
    'pred.properties': 'properties',
    'pred.priceCorrection': 'Price Correction',
    'pred.feedbackSent': 'Feedback sent — thank you!',
    'pred.actualPricePlaceholder': 'Actual price (min Rp 10,000,000)...',
    'pred.send': 'Send',
    'pred.feedbackMinPrice': 'Minimum price is Rp 10,000,000',
    'pred.feedbackError': 'Failed to send feedback.',
    'pred.comparableTitle': 'Comparable Properties',
    'pred.similarity': 'similar',

    'chat.title': 'AI Assistant',
    'chat.subtitle': 'Ask about property',
    'chat.location': 'Location',
    'chat.welcome': 'Hi! I can help answer questions about estimated prices and the property market in Depok.',
    'chat.placeholder': 'Type your question...',
    'chat.error': 'An error occurred.',
    'chat.quick1': 'How is the price calculated?',
    'chat.quick2': 'What is the upper-middle segment?',
    'chat.quick3': 'What factors affect price?',

    'history.pageTitle': 'Prediction History',
    'history.pageSubtitle': 'List of predictions made in this session',
    'history.emptyTitle': 'No prediction history yet',
    'history.emptySubtitle': 'Run a property analysis first',
    'history.delete': 'Delete',

    'analytics.pageTitle': 'Area Analytics',
    'analytics.pageSubtitle': 'Market summary by location based on training data',
    'analytics.trendPerQuarter': '/ quarter',
    'analytics.avgPricePerM2': 'Avg price/m²',
    'analytics.propertyData': 'Property data',
    'analytics.unit': 'units',
    'analytics.dominantSegment': 'Dominant segment',
    'analytics.errorRate': 'error rate',
    'analytics.noTrend': 'N/A',
    'analytics.catatan': 'Data analyzed dynamically based on the classification model.',
    'analytics.loadError': 'Failed to load analytics',
  },
  zh: {
    'app.langLabel': '语言',
    'nav.menu': '菜单',
    'nav.prediksi': '预测',
    'nav.riwayat': '历史',
    'nav.analitik': '分析',
    'app.modelActive': '模型运行中',
    'app.estimasiProperti': '房产估值',

    'pred.pageTitle': '房价预测',
    'pred.pageSubtitle': '使用机器学习模型估算价格、分层与聚类',
    'pred.parameterTitle': '房产参数',
    'pred.bedroom': '卧室',
    'pred.bathroom': '浴室',
    'pred.garage': '车库',
    'pred.landArea': '土地面积 (m²)',
    'pred.buildingArea': '建筑面积 (m²)',
    'pred.location': '位置',
    'pred.analyzing': '分析中...',
    'pred.analyze': '分析房产',
    'pred.errorGeneric': '发生错误。',
    'pred.resultTitle': '分析结果',
    'pred.estimatedPrice': '估算价格',
    'pred.segment': '分层',
    'pred.cluster': '聚类',
    'pred.modelUsed': '使用模型：',
    'pred.probability': '概率',
    'pred.clusterMedian': '中位数',
    'pred.clusterDominantLocation': '主要区域',
    'pred.clusterLandMedian': '土地面积中位数',
    'pred.clusterRange': '范围',
    'pred.clusterDataCount': '数据量',
    'pred.properties': '套房产',
    'pred.priceCorrection': '价格修正',
    'pred.feedbackSent': '反馈已发送 — 感谢！',
    'pred.actualPricePlaceholder': '实际价格（最低 Rp 10,000,000）...',
    'pred.send': '发送',
    'pred.feedbackMinPrice': '最低价格为 Rp 10,000,000',
    'pred.feedbackError': '发送反馈失败。',
    'pred.comparableTitle': '可比房源',
    'pred.similarity': '相似',

    'chat.title': 'AI 助手',
    'chat.subtitle': '咨询房产问题',
    'chat.location': '位置',
    'chat.welcome': '你好！我可以帮助你解答有关德博房产价格估算和市场的问题。',
    'chat.placeholder': '输入你的问题...',
    'chat.error': '发生错误。',
    'chat.quick1': '房价是如何计算的？',
    'chat.quick2': '什么是中高端细分市场？',
    'chat.quick3': '哪些因素会影响房价？',

    'history.pageTitle': '预测历史',
    'history.pageSubtitle': '本次会话中的预测记录',
    'history.emptyTitle': '暂无预测历史',
    'history.emptySubtitle': '请先进行房产分析',
    'history.delete': '删除',

    'analytics.pageTitle': '区域分析',
    'analytics.pageSubtitle': '基于训练数据的分区域市场概览',
    'analytics.trendPerQuarter': '/ 季度',
    'analytics.avgPricePerM2': '平均价格/m²',
    'analytics.propertyData': '房产数据',
    'analytics.unit': '套',
    'analytics.dominantSegment': '主导分层',
    'analytics.errorRate': '误差率',
    'analytics.noTrend': '暂无',
    'analytics.catatan': '数据基于分类模型动态分析。',
    'analytics.loadError': '加载分析数据失败',
  },
}

export function normalizeLang(value?: string | null): Lang {
  if (value === 'id' || value === 'en' || value === 'zh') return value
  return 'id'
}

export function getStoredLang(): Lang {
  if (typeof window === 'undefined') return 'id'
  return normalizeLang(window.localStorage.getItem(LANG_KEY))
}

export function setStoredLang(lang: Lang): void {
  if (typeof window === 'undefined') return
  window.localStorage.setItem(LANG_KEY, lang)
}

export function getLocale(lang: Lang): string {
  if (lang === 'en') return 'en-US'
  if (lang === 'zh') return 'zh-CN'
  return 'id-ID'
}

export function t(lang: Lang, key: string): string {
  return messages[lang]?.[key] ?? messages.id[key] ?? key
}
