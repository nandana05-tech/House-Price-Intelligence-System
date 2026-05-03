import clsx from 'clsx'

/* ── Card ── */
export function Card({ children, className }: { children: React.ReactNode; className?: string }) {
  return (
    <div className={clsx('bg-white border border-stone-200 rounded-xl p-5', className)}>
      {children}
    </div>
  )
}

/* ── Section title ── */
export function SectionTitle({ children }: { children: React.ReactNode }) {
  return (
    <p className="text-[10px] font-semibold uppercase tracking-widest text-stone-400 mb-4">
      {children}
    </p>
  )
}

/* ── Input ── */
export function Input({
  label, type = 'text', value, onChange, placeholder, min, max,
}: {
  label: string; type?: string; value: string | number; onChange: (v: string) => void;
  placeholder?: string; min?: number; max?: number;
}) {
  return (
    <div className="flex flex-col gap-1.5">
      <label className="text-[11px] font-medium text-stone-500">{label}</label>
      <input
        type={type}
        value={value}
        min={min}
        max={max}
        placeholder={placeholder}
        onChange={(e) => onChange(e.target.value)}
        className="w-full px-3 py-2 text-[13px] bg-stone-50 border border-stone-200 rounded-lg text-stone-900 placeholder-stone-300 focus:outline-none focus:border-amber-400 focus:ring-1 focus:ring-amber-400/30 transition-all"
      />
    </div>
  )
}

/* ── Select ── */
export function Select({
  label, value, onChange, options,
}: {
  label: string; value: string; onChange: (v: string) => void; options: string[];
}) {
  return (
    <div className="flex flex-col gap-1.5">
      <label className="text-[11px] font-medium text-stone-500">{label}</label>
      <select
        value={value}
        onChange={(e) => onChange(e.target.value)}
        className="w-full px-3 py-2 text-[13px] bg-stone-50 border border-stone-200 rounded-lg text-stone-900 focus:outline-none focus:border-amber-400 focus:ring-1 focus:ring-amber-400/30 transition-all appearance-none cursor-pointer"
      >
        {options.map((o) => <option key={o}>{o}</option>)}
      </select>
    </div>
  )
}

/* ── Button ── */
export function Button({
  children, onClick, loading, variant = 'primary', className,
}: {
  children: React.ReactNode; onClick?: () => void; loading?: boolean;
  variant?: 'primary' | 'secondary' | 'ghost'; className?: string;
}) {
  return (
    <button
      onClick={onClick}
      disabled={loading}
      className={clsx(
        'inline-flex items-center justify-center gap-2 rounded-lg text-[13px] font-medium transition-all duration-150 disabled:opacity-50 disabled:cursor-not-allowed',
        variant === 'primary'   && 'bg-amber-500 hover:bg-amber-600 active:scale-[0.98] text-white px-4 py-2.5',
        variant === 'secondary' && 'bg-stone-100 hover:bg-stone-200 active:scale-[0.98] text-stone-700 px-4 py-2.5',
        variant === 'ghost'     && 'hover:bg-stone-50 text-stone-500 hover:text-stone-800 px-3 py-2',
        className,
      )}
    >
      {loading && (
        <svg className="animate-spin w-3.5 h-3.5" viewBox="0 0 24 24" fill="none">
          <circle className="opacity-25" cx="12" cy="12" r="10" stroke="currentColor" strokeWidth="4"/>
          <path className="opacity-75" fill="currentColor" d="M4 12a8 8 0 018-8v8z"/>
        </svg>
      )}
      {children}
    </button>
  )
}

/* ── Metric card ── */
export function Metric({
  label, value, sub, accent,
}: {
  label: string; value: string; sub?: string; accent?: boolean;
}) {
  return (
    <div className={clsx(
      'rounded-lg p-4 flex flex-col gap-1',
      accent ? 'bg-amber-50 border border-amber-200' : 'bg-stone-50 border border-stone-100',
    )}>
      <span className={clsx('text-[10px] font-semibold uppercase tracking-wider', accent ? 'text-amber-600' : 'text-stone-400')}>
        {label}
      </span>
      <span className={clsx('text-xl font-semibold leading-none', accent ? 'text-amber-700' : 'text-stone-900')}>
        {value}
      </span>
      {sub && <span className={clsx('text-[11px]', accent ? 'text-amber-500' : 'text-stone-400')}>{sub}</span>}
    </div>
  )
}

/* ── Pill badge ── */
export function Pill({ children, color = 'stone' }: { children: React.ReactNode; color?: 'stone' | 'green' | 'blue' | 'amber' | 'red' }) {
  const map = {
    stone: 'bg-stone-100 text-stone-600',
    green: 'bg-emerald-50 text-emerald-700 border border-emerald-200',
    blue:  'bg-sky-50 text-sky-700 border border-sky-200',
    amber: 'bg-amber-50 text-amber-700 border border-amber-200',
    red:   'bg-red-50 text-red-700 border border-red-200',
  }
  return (
    <span className={clsx('inline-flex items-center px-2.5 py-0.5 rounded-full text-[11px] font-medium', map[color])}>
      {children}
    </span>
  )
}

/* ── Skeleton loader ── */
export function Skeleton({ className }: { className?: string }) {
  return <div className={clsx('skeleton rounded-md', className)} />
}

/* ── Page header ── */
export function PageHeader({ title, subtitle }: { title: string; subtitle?: string }) {
  return (
    <div className="px-8 py-6 border-b border-stone-100 bg-white">
      <h1 className="text-[18px] font-semibold text-stone-900">{title}</h1>
      {subtitle && <p className="text-[13px] text-stone-400 mt-0.5">{subtitle}</p>}
    </div>
  )
}
