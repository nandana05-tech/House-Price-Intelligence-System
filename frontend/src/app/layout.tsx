'use client'

import './globals.css'
import Link from 'next/link'
import { usePathname } from 'next/navigation'
import clsx from 'clsx'

const navItems = [
  {
    href: '/',
    label: 'Prediksi',
    icon: (
      <svg width="16" height="16" viewBox="0 0 16 16" fill="none">
        <path d="M2 14L5.5 9.5L8.5 12L11.5 7L14 10" stroke="currentColor" strokeWidth="1.5" strokeLinecap="round" strokeLinejoin="round"/>
        <path d="M2 2h12" stroke="currentColor" strokeWidth="1.5" strokeLinecap="round"/>
        <path d="M2 2v12" stroke="currentColor" strokeWidth="1.5" strokeLinecap="round"/>
      </svg>
    ),
  },
  {
    href: '/riwayat',
    label: 'Riwayat',
    icon: (
      <svg width="16" height="16" viewBox="0 0 16 16" fill="none">
        <circle cx="8" cy="8" r="6" stroke="currentColor" strokeWidth="1.5"/>
        <path d="M8 5v3l2 1.5" stroke="currentColor" strokeWidth="1.5" strokeLinecap="round" strokeLinejoin="round"/>
      </svg>
    ),
  },
  {
    href: '/analitik',
    label: 'Analitik',
    icon: (
      <svg width="16" height="16" viewBox="0 0 16 16" fill="none">
        <rect x="2" y="9" width="3" height="5" rx="1" fill="currentColor" opacity="0.6"/>
        <rect x="6.5" y="5" width="3" height="9" rx="1" fill="currentColor" opacity="0.8"/>
        <rect x="11" y="2" width="3" height="12" rx="1" fill="currentColor"/>
      </svg>
    ),
  },
]

export default function RootLayout({ children }: { children: React.ReactNode }) {
  const pathname = usePathname()

  return (
    <html lang="id">
      <body className="flex h-screen overflow-hidden bg-stone-50">
        {/* Sidebar */}
        <aside className="w-[200px] flex-shrink-0 flex flex-col border-r border-stone-200 bg-white">
          {/* Logo */}
          <div className="px-5 py-5 border-b border-stone-100">
            <div className="flex items-center gap-2.5">
              <div className="w-7 h-7 rounded-md bg-amber-500 flex items-center justify-center flex-shrink-0">
                <svg width="14" height="14" viewBox="0 0 14 14" fill="none">
                  <path d="M2 11L5 6.5L7.5 9L10 5L12 7.5" stroke="white" strokeWidth="1.5" strokeLinecap="round" strokeLinejoin="round"/>
                </svg>
              </div>
              <div>
                <div className="text-[13px] font-semibold text-stone-900 leading-none">PropValAI</div>
                <div className="text-[10px] text-stone-400 mt-0.5">Estimasi Properti</div>
              </div>
            </div>
          </div>

          {/* Nav */}
          <nav className="flex-1 py-3 px-2">
            <div className="text-[10px] font-medium text-stone-400 px-3 mb-2 uppercase tracking-widest">Menu</div>
            {navItems.map((item) => {
              const active = pathname === item.href
              return (
                <Link
                  key={item.href}
                  href={item.href}
                  className={clsx(
                    'flex items-center gap-2.5 px-3 py-2 rounded-lg text-[13px] transition-all duration-150 mb-0.5',
                    active
                      ? 'bg-amber-50 text-amber-700 font-medium'
                      : 'text-stone-500 hover:bg-stone-50 hover:text-stone-800'
                  )}
                >
                  <span className={clsx(active ? 'text-amber-600' : 'text-stone-400')}>{item.icon}</span>
                  {item.label}
                </Link>
              )
            })}
          </nav>

          {/* Footer status */}
          <div className="px-5 py-4 border-t border-stone-100">
            <div className="flex items-center gap-2">
              <span className="w-1.5 h-1.5 rounded-full bg-emerald-400 animate-pulse-slow"/>
              <span className="text-[11px] text-stone-400">Model aktif</span>
            </div>
          </div>
        </aside>

        {/* Main */}
        <main className="flex-1 overflow-auto">
          {children}
        </main>
      </body>
    </html>
  )
}
