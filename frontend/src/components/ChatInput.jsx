import { useState } from 'react'

// Text input bar at the bottom of the chat.
// Pressing Enter (without Shift) submits the form.
// Shift+Enter inserts a newline.
export default function ChatInput({ onSend, disabled }) {
  const [text, setText] = useState('')

  function handleSubmit(e) {
    e.preventDefault()
    const trimmed = text.trim()
    if (!trimmed || disabled) return
    onSend(trimmed)
    setText('')
  }

  function handleKeyDown(e) {
    if (e.key === 'Enter' && !e.shiftKey) {
      handleSubmit(e)
    }
  }

  return (
    <div className="input-area">
      <form className="input-form" onSubmit={handleSubmit}>
        <textarea
          className="input-box"
          rows={1}
          placeholder="Ask a legal question about Pakistani law..."
          value={text}
          onChange={e => setText(e.target.value)}
          onKeyDown={handleKeyDown}
          disabled={disabled}
        />
        <button
          type="submit"
          className="send-btn"
          disabled={disabled || !text.trim()}
        >
          {disabled ? '...' : 'Send'}
        </button>
      </form>
      <p className="input-hint">
        This chatbot provides legal information, not formal legal advice. Consult a licensed advocate for serious matters.
      </p>
    </div>
  )
}
