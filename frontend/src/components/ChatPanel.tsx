'use client'

import { useState, useRef, useEffect } from 'react'
import { api, type ChatMessage } from '@/lib/api'

interface Message {
  role:        'user' | 'bot'
  content:     string
  tools_used?: string[]
}

function FormattedMessage({ content }: { content: string }) {
  const lines = content.split('\n').filter(l => l.trim() !== '')
  return (
    <div className="space-y-1.5">
      {lines.map((line, i) => {
        const parts = line.split(/(\*\*[^*]+\*\*)/g)
        return (
          <p key={i}>
            {parts.map((part, j) =>
              part.startsWith('**') && part.endsWith('**')
                ? <strong key={j} className="font-semibold">{part.slice(2, -2)}</strong>
                : part
            )}
          </p>
        )
      })}
    </div>
  )
}

const QUICK_PROMPTS = [
  'Bagaimana cara harga dihitung?',
  'Apa itu segmen menengah atas?',
  'Faktor apa yang mempengaruhi harga?',
]

export default function ChatPanel({
  lokasi,
  hargaPrediksi,
}: {
  lokasi?:        string
  hargaPrediksi?: number
}) {
  const [messages, setMessages] = useState<Message[]>([
    {
      role:    'bot',
      content: 'Halo! Saya siap membantu menjawab pertanyaan seputar estimasi harga dan pasar properti di Depok.',
    },
  ])
  // History untuk dikirim ke backend (hanya role user/assistant)
  const [history, setHistory] = useState<ChatMessage[]>([])
  const [input,   setInput]   = useState('')
  const [loading, setLoading] = useState(false)
  const bottomRef = useRef<HTMLDivElement>(null)

  useEffect(() => {
    bottomRef.current?.scrollIntoView({ behavior: 'smooth' })
  }, [messages])

  async function send(text?: string) {
    const msg = (text ?? input).trim()
    if (!msg || loading) return
    setInput('')

    // Tambah ke UI
    setMessages(prev => [...prev, { role: 'user', content: msg }])

    // Tambah ke history untuk backend
    const newHistory: ChatMessage[] = [...history, { role: 'user', content: msg }]

    setLoading(true)
    try {
      // Kirim pesan + history (tanpa pesan terakhir karena sudah ada di message)
      const res = await api.chat(msg, history)

      setMessages(prev => [...prev, {
        role:       'bot',
        content:    res.reply,
        tools_used: res.tools_used,
      }])

      // Update history dengan pesan user + reply assistant
      setHistory([...newHistory, { role: 'assistant', content: res.reply }])

    } catch (e: unknown) {
      const errMsg = e instanceof Error ? e.message : 'Terjadi kesalahan.'
      setMessages(prev => [...prev, { role: 'bot', content: `Error: ${errMsg}` }])
    } finally {
      setLoading(false)
    }
  }

  return (
    <div className="flex flex-col h-full">
      {/* Header */}
      <div className="px-4 py-4 border-b border-stone-100">
        <p className="text-[10px] font-semibold uppercase tracking-widest text-stone-400 mb-0.5">Asisten AI</p>
        <p className="text-[13px] font-medium text-stone-800">Tanya soal properti</p>
        {lokasi && <p className="text-[11px] text-stone-400 mt-0.5">Lokasi: {lokasi}</p>}
      </div>

      {/* Messages */}
      <div className="flex-1 overflow-y-auto px-4 py-4 flex flex-col gap-3">
        {messages.map((m, i) => (
          <div key={i} className={m.role === 'user' ? 'flex justify-end' : 'flex justify-start'}>
            <div
              className={
                m.role === 'user'
                  ? 'bg-amber-500 text-white rounded-2xl rounded-tr-sm px-3.5 py-2.5 text-[12px] leading-relaxed max-w-[85%]'
                  : 'bg-stone-100 text-stone-700 rounded-2xl rounded-tl-sm px-3.5 py-2.5 text-[12px] leading-relaxed max-w-[90%]'
              }
            >
              <FormattedMessage content={m.content} />
              {m.tools_used && m.tools_used.length > 0 && (
                <div className="mt-1.5 flex flex-wrap gap-1">
                  {m.tools_used.map(t => (
                    <span key={t} className="text-[10px] bg-stone-200 text-stone-500 px-1.5 py-0.5 rounded-full">
                      {t}
                    </span>
                  ))}
                </div>
              )}
            </div>
          </div>
        ))}

        {loading && (
          <div className="flex justify-start">
            <div className="bg-stone-100 rounded-2xl rounded-tl-sm px-4 py-3 flex gap-1 items-center">
              <span className="w-1.5 h-1.5 bg-stone-400 rounded-full animate-bounce" style={{ animationDelay: '0ms' }} />
              <span className="w-1.5 h-1.5 bg-stone-400 rounded-full animate-bounce" style={{ animationDelay: '150ms' }} />
              <span className="w-1.5 h-1.5 bg-stone-400 rounded-full animate-bounce" style={{ animationDelay: '300ms' }} />
            </div>
          </div>
        )}
        <div ref={bottomRef} />
      </div>

      {/* Quick prompts */}
      {messages.length < 3 && (
        <div className="px-4 pb-3 flex flex-col gap-1.5">
          {QUICK_PROMPTS.map(q => (
            <button
              key={q}
              onClick={() => send(q)}
              className="text-left text-[11px] text-stone-500 bg-stone-50 hover:bg-stone-100 border border-stone-200 rounded-lg px-3 py-2 transition-colors leading-snug"
            >
              {q}
            </button>
          ))}
        </div>
      )}

      {/* Input */}
      <div className="px-4 pb-4 pt-2 border-t border-stone-100">
        <div className="flex gap-2">
          <input
            value={input}
            onChange={e => setInput(e.target.value)}
            onKeyDown={e => { if (e.key === 'Enter') send() }}
            placeholder="Tulis pertanyaan..."
            className="flex-1 px-3 py-2 text-[12px] bg-stone-50 border border-stone-200 rounded-lg text-stone-900 placeholder-stone-300 focus:outline-none focus:border-amber-400 focus:ring-1 focus:ring-amber-400/30 transition-all"
          />
          <button
            onClick={() => send()}
            disabled={loading || !input.trim()}
            className="w-8 h-8 flex items-center justify-center bg-amber-500 hover:bg-amber-600 disabled:opacity-40 rounded-lg transition-all active:scale-95"
          >
            <svg width="14" height="14" viewBox="0 0 14 14" fill="none">
              <path d="M12 7L2 2l2 5-2 5 10-5z" fill="white"/>
            </svg>
          </button>
        </div>
      </div>
    </div>
  )
}