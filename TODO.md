# TODO - Multilingual Chat & UI (ID/EN/ZH)

- [x] Add backend language support in `api/chat_endpoint.py`
  - [x] Extend ChatRequest with `language`
  - [x] Add language normalization/fallback (`id|en|zh`)
  - [x] Build dynamic system prompt by language
  - [x] Keep tool-calling + RAG flow intact

- [x] Add frontend i18n foundation
  - [x] Create `frontend/src/lib/i18n.ts`
  - [x] Add dictionary keys for `id`, `en`, `zh`
  - [x] Add helpers for language persistence and translation

- [x] Update API client for chat language payload
  - [x] Edit `frontend/src/lib/api.ts` to accept `language` in `api.chat(...)`

- [x] Update chat UI for language switching
  - [x] Edit `frontend/src/components/ChatPanel.tsx`
  - [x] Add language selector (ID/EN/中文)
  - [x] Localize quick prompts, placeholders, errors, loading text
  - [x] Send selected language to backend

- [x] Localize app pages
  - [x] Edit `frontend/src/app/layout.tsx`
  - [x] Edit `frontend/src/app/page.tsx`
  - [x] Edit `frontend/src/app/riwayat/page.tsx`
  - [x] Edit `frontend/src/app/analitik/page.tsx`

- [ ] Validate
  - [ ] Run frontend type/build check
  - [ ] Confirm chat request contains `language`
  - [ ] Confirm responses and UI text switch correctly for ID/EN/ZH
