import React, { useState } from 'react';
import DOMPurify from 'dompurify';

export interface AuditComment {
  id: number;
  author: string;
  author_url: string;
  body: string;
}

interface Props {
  comments: AuditComment[];
  onAddComment: (comment: { author: string; author_url: string; body: string }) => Promise<void>;
}

export const AuditCommentFeed: React.FC<Props> = ({ comments, onAddComment }) => {
  const [author, setAuthor] = useState('');
  const [authorUrl, setAuthorUrl] = useState('');
  const [body, setBody] = useState('');
  const [isSubmitting, setIsSubmitting] = useState(false);

  // ---------------------------------------------------------------------------
  // Exercise 3 - Task A: Protocol Allowlisting for URL / Link Sinks
  // ---------------------------------------------------------------------------
  // In React, JSX prevents basic script injection in text nodes, but
  // <a href={url}> is an unescaped sink! Passing "javascript:..." executes arbitrary JS!
  //
  // TODO(human):
  // 1. Validate that the URL starts with a safe protocol ('http://', 'https://', 'mailto:').
  // 2. Reject 'javascript:', 'data:', or 'vbscript:'.
  // 3. Return a safe sanitized URL or '#' if invalid.
  const sanitizeUrl = (rawUrl: string): string => {
    // NAIVE / UNHARDENED STARTER (Directly accepts javascript: URLs!):
    return rawUrl;
  };

  // ---------------------------------------------------------------------------
  // Exercise 3 - Task B: Safe Rich-Text HTML Sanitization with DOMPurify
  // ---------------------------------------------------------------------------
  // Rich text comments may contain formatting tags (<b>, <i>, <code>),
  // but attackers can inject <img src=x onerror=...>, <script>, or inline event handlers.
  //
  // TODO(human):
  // 1. Sanitize rawHtml using DOMPurify.sanitize(rawHtml, {
  //      ALLOWED_TAGS: ['b', 'i', 'em', 'strong', 'code', 'pre', 'p', 'br'],
  //      ALLOWED_ATTR: ['class'],
  //    });
  // 2. Return the sanitized HTML string.
  const sanitizeHtmlBody = (rawHtml: string): string => {
    // NAIVE / UNHARDENED STARTER (Directly returns untrusted HTML - vulnerable to XSS!):
    return rawHtml;
  };

  const handleSubmit = async (e: React.FormEvent) => {
    e.preventDefault();
    if (!author || !body) return;
    setIsSubmitting(true);
    try {
      await onAddComment({ author, author_url: authorUrl || 'https://internal.audit', body });
      setAuthor('');
      setAuthorUrl('');
      setBody('');
    } finally {
      setIsSubmitting(false);
    }
  };

  return (
    <div style={{ marginTop: 24, padding: 18, background: '#1e293b', borderRadius: 8, border: '1px solid #334155' }}>
      <h3 style={{ color: '#38bdf8', marginTop: 0 }}>Treasury Audit & Compliance Feed</h3>
      <p style="color: #94a3b8; font-size: 13px;" style={{ color: '#94a3b8', fontSize: 13 }}>
        Examines React XSS vulnerability vectors: unsafe <code>href</code> sinks and unsanitized rich-text bodies.
      </p>

      {/* Comment Form */}
      <form onSubmit={handleSubmit} style={{ marginBottom: 20, display: 'flex', flexDirection: 'column', gap: 10 }}>
        <div style={{ display: 'flex', gap: 10 }}>
          <input
            type="text"
            placeholder="Auditor Name"
            value={author}
            onChange={(e) => setAuthor(e.target.value)}
            style={{ flex: 1, padding: 8, background: '#0f172a', border: '1px solid #475569', color: '#fff', borderRadius: 4 }}
            required
          />
          <input
            type="text"
            placeholder="Auditor Profile URL (Sink A: try javascript:alert(1))"
            value={authorUrl}
            onChange={(e) => setAuthorUrl(e.target.value)}
            style={{ flex: 2, padding: 8, background: '#0f172a', border: '1px solid #475569', color: '#fff', borderRadius: 4 }}
          />
        </div>
        <textarea
          placeholder="Audit Comments (Sink B: try <img src=x onerror=alert('XSS')>)"
          value={body}
          onChange={(e) => setBody(e.target.value)}
          rows={3}
          style={{ width: '100%', padding: 8, background: '#0f172a', border: '1px solid #475569', color: '#fff', borderRadius: 4, boxSizing: 'border-box' }}
          required
        />
        <button
          type="submit"
          disabled={isSubmitting}
          style={{
            alignSelf: 'flex-start',
            padding: '8px 16px',
            background: '#2563eb',
            color: '#fff',
            border: 'none',
            borderRadius: 4,
            cursor: 'pointer',
            fontWeight: 600,
          }}
        >
          {isSubmitting ? 'Posting...' : 'Post Audit Entry'}
        </button>
      </form>

      {/* Comments List */}
      <div style={{ display: 'flex', flexDirection: 'column', gap: 12 }}>
        {comments.map((comment) => {
          const safeUrl = sanitizeUrl(comment.author_url);
          const safeBodyHtml = sanitizeHtmlBody(comment.body);

          return (
            <div
              key={comment.id}
              style={{
                padding: 12,
                background: '#0f172a',
                border: '1px solid #334155',
                borderRadius: 6,
              }}
            >
              <div style={{ display: 'flex', justifyContent: 'space-between', marginBottom: 6 }}>
                <strong>{comment.author}</strong>
                {/* Vulnerability Sink A: <a href> without protocol check */}
                <a
                  href={safeUrl}
                  target="_blank"
                  rel="noreferrer"
                  style={{ color: '#60a5fa', fontSize: 13, textDecoration: 'underline' }}
                >
                  Verify Auditor Profile
                </a>
              </div>

              {/* Vulnerability Sink B: dangerouslySetInnerHTML without DOMPurify */}
              <div
                style={{ fontSize: 14, color: '#e2e8f0', lineHeight: 1.5 }}
                dangerouslySetInnerHTML={{ __html: safeBodyHtml }}
              />
            </div>
          );
        })}
      </div>
    </div>
  );
};
