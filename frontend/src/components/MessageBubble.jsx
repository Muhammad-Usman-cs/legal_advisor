// A single chat message. role is either 'user' or 'assistant'.
// white-space: pre-wrap in CSS preserves newlines in AI responses.
export default function MessageBubble({ role, content }) {
  const isUser = role === 'user'

  return (
    <div className={`message ${isUser ? 'message-user' : 'message-assistant'}`}>
      <div className="message-avatar">
        {isUser ? '👤' : '⚖️'}
      </div>
      <div className="message-bubble">
        <p className="message-content">{content}</p>
      </div>
    </div>
  )
}
