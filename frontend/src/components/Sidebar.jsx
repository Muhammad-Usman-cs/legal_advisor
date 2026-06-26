// Left sidebar: new chat button, conversation list, logout button.
export default function Sidebar({
  conversations,
  activeConvId,
  onSelect,
  onNew,
  onDelete,
  user,
  onLogout,
}) {
  function handleDelete(e, id) {
    // Stop the click from also triggering onSelect for this conversation
    e.stopPropagation()
    if (confirm('Delete this conversation?')) {
      onDelete(id)
    }
  }

  return (
    <aside className="sidebar">
      <div className="sidebar-header">
        <span className="sidebar-logo">⚖️</span>
        <span className="sidebar-title">Legal Advisor</span>
      </div>

      <button className="new-chat-btn" onClick={onNew}>
        + New Conversation
      </button>

      <nav className="conversations-list">
        {conversations.length === 0 && (
          <p className="no-convs">No conversations yet</p>
        )}
        {conversations.map(conv => (
          <div
            key={conv.id}
            className={`conv-item ${conv.id === activeConvId ? 'conv-active' : ''}`}
            onClick={() => onSelect(conv.id)}
          >
            <span className="conv-title">{conv.title}</span>
            <button
              className="conv-delete"
              onClick={e => handleDelete(e, conv.id)}
              title="Delete"
            >
              ✕
            </button>
          </div>
        ))}
      </nav>

      <div className="sidebar-footer">
        <span className="sidebar-user">👤 {user?.username ?? '...'}</span>
        <button className="logout-btn" onClick={onLogout}>Logout</button>
      </div>
    </aside>
  )
}
