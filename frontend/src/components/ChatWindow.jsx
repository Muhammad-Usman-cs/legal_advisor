import { useEffect, useRef } from 'react'
import MessageBubble from './MessageBubble'

// Displays the list of messages. Auto-scrolls to the bottom on new messages.
export default function ChatWindow({ messages, isLoading }) {
  const bottomRef = useRef(null)

  useEffect(() => {
    bottomRef.current?.scrollIntoView({ behavior: 'smooth' })
  }, [messages, isLoading])

  if (messages.length === 0 && !isLoading) {
    return (
      <div className="chat-window empty">
        <div className="empty-state">
          <span className="empty-icon">⚖️</span>
          <h2>Pakistani Legal Advisor</h2>
          <p>Ask me anything about Pakistani law — property disputes, criminal law, family law, labour rights, and more.</p>
        </div>
      </div>
    )
  }

  return (
    <div className="chat-window">
      {messages.map((msg, i) => (
        <MessageBubble key={msg.id ?? i} role={msg.role} content={msg.content} />
      ))}

      {isLoading && (
        <div className="message message-assistant">
          <div className="message-avatar">⚖️</div>
          <div className="message-bubble loading-bubble">
            <span className="dot" />
            <span className="dot" />
            <span className="dot" />
          </div>
        </div>
      )}

      <div ref={bottomRef} />
    </div>
  )
}
