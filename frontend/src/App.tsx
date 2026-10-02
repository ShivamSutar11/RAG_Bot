import React, { useState, useEffect, useRef } from 'react';
import axios from 'axios';
import ReactMarkdown from 'react-markdown';
import { Upload, Trash2, Send, FileText, Loader2, RefreshCw } from 'lucide-react';

const API_URL = import.meta.env.VITE_API_URL || '/api';

interface Document {
  document_id: string;
  filename: string;
}

interface Message {
  role: 'user' | 'assistant';
  content: string;
  sourceDocument?: string;
}

function App() {
  const [documents, setDocuments] = useState<Document[]>([]);
  const [activeDocId, setActiveDocId] = useState<string | null>(null);
  const [messages, setMessages] = useState<Message[]>([]);
  const [input, setInput] = useState('');
  const [isUploading, setIsUploading] = useState(false);
  const [isAnswering, setIsAnswering] = useState(false);
  
  const messagesEndRef = useRef<HTMLDivElement>(null);

  useEffect(() => {
    fetchDocuments();
  }, []);

  useEffect(() => {
    messagesEndRef.current?.scrollIntoView({ behavior: 'smooth' });
  }, [messages, isAnswering]);

  const fetchDocuments = async () => {
    try {
      const res = await axios.get(`${API_URL}/documents`);
      setDocuments(res.data);
      if (res.data.length > 0 && !activeDocId) {
        setActiveDocId(res.data[0].document_id);
      }
    } catch (err) {
      console.error('Failed to fetch documents', err);
    }
  };

  const handleFileUpload = async (e: React.ChangeEvent<HTMLInputElement>) => {
    const files = e.target.files;
    if (!files || files.length === 0) return;

    const file = files[0];
    const formData = new FormData();
    formData.append('file', file);

    setIsUploading(true);
    try {
      const res = await axios.post(`${API_URL}/documents/upload`, formData);
      await fetchDocuments();
      setActiveDocId(res.data.document_id);
    } catch (err) {
      console.error('Upload failed', err);
      alert('Upload failed. Only PDF/TXT/DOCX are supported.');
    } finally {
      setIsUploading(false);
      if (e.target) e.target.value = '';
    }
  };

  const handleDeleteDoc = async (id: string) => {
    try {
      await axios.delete(`${API_URL}/documents/${id}`);
      const newDocs = documents.filter(d => d.document_id !== id);
      setDocuments(newDocs);
      if (activeDocId === id) {
        setActiveDocId(newDocs.length > 0 ? newDocs[0].document_id : null);
      }
    } catch (err) {
      console.error('Delete failed', err);
    }
  };

  const handleSend = async () => {
    if (!input.trim() || !activeDocId) return;

    const userMessage = input.trim();
    setInput('');
    setMessages(prev => [...prev, { role: 'user', content: userMessage }]);
    setIsAnswering(true);

    try {
      const res = await axios.post(`${API_URL}/chat`, {
        document_id: activeDocId,
        question: userMessage
      });
      setMessages(prev => [...prev, {
        role: 'assistant',
        content: res.data.answer,
        sourceDocument: res.data.source_document
      }]);
    } catch (err: any) {
      console.error('Chat failed', err);
      setMessages(prev => [...prev, {
        role: 'assistant',
        content: `Error: ${err.response?.data?.detail || err.message}`
      }]);
    } finally {
      setIsAnswering(false);
    }
  };

  const handleKeyDown = (e: React.KeyboardEvent) => {
    if (e.key === 'Enter' && !e.shiftKey) {
      e.preventDefault();
      handleSend();
    }
  };

  const clearChat = () => {
    setMessages([]);
  };

  return (
    <div className="flex h-screen bg-gray-900 text-gray-100 font-sans">
      {/* Sidebar */}
      <div className="w-72 bg-gray-950 border-r border-gray-800 flex flex-col">
        <div className="p-4 border-b border-gray-800">
          <h1 className="text-xl font-bold text-orange-500 flex items-center gap-2">
            <RefreshCw className="w-6 h-6" />
            RAG_Bot
          </h1>
        </div>
        
        <div className="flex-1 overflow-y-auto p-4 space-y-4">
          <div>
            <h2 className="text-sm font-semibold text-gray-400 mb-2 uppercase tracking-wider">Uploaded Documents</h2>
            
            <div className="space-y-2">
              {documents.length === 0 && (
                <p className="text-gray-500 text-sm italic">No documents uploaded.</p>
              )}
              {documents.map(doc => (
                <div 
                  key={doc.document_id}
                  onClick={() => setActiveDocId(doc.document_id)}
                  className={`flex items-center justify-between p-2 rounded cursor-pointer transition-colors ${
                    activeDocId === doc.document_id ? 'bg-orange-900/30 border border-orange-500/50 text-orange-100' : 'hover:bg-gray-800 border border-transparent'
                  }`}
                >
                  <div className="flex items-center gap-2 overflow-hidden">
                    <FileText className="w-4 h-4 flex-shrink-0 opacity-70" />
                    <span className="truncate text-sm">{doc.filename}</span>
                  </div>
                  <button 
                    onClick={(e) => { e.stopPropagation(); handleDeleteDoc(doc.document_id); }}
                    className="p-1 hover:bg-gray-700 rounded text-gray-400 hover:text-red-400"
                    title="Delete document"
                  >
                    <Trash2 className="w-4 h-4" />
                  </button>
                </div>
              ))}
            </div>
          </div>
        </div>

        <div className="p-4 border-t border-gray-800">
          <label className="flex items-center justify-center gap-2 w-full px-4 py-2 bg-gray-800 hover:bg-gray-700 border border-gray-700 rounded-md cursor-pointer transition-colors">
            {isUploading ? <Loader2 className="w-4 h-4 animate-spin" /> : <Upload className="w-4 h-4" />}
            <span className="text-sm font-medium">{isUploading ? 'Uploading...' : 'Upload Document'}</span>
            <input type="file" className="hidden" accept=".pdf,.txt,.docx" onChange={handleFileUpload} disabled={isUploading} />
          </label>
        </div>
      </div>

      {/* Main Chat Area */}
      <div className="flex-1 flex flex-col bg-gray-900">
        <div className="h-14 border-b border-gray-800 flex items-center justify-between px-6 bg-gray-900/50 backdrop-blur">
          <div className="flex items-center gap-2 text-sm text-gray-300">
            <span className="font-medium text-gray-400">Active Document:</span>
            {activeDocId ? (
              <span className="px-2 py-1 bg-gray-800 rounded text-orange-400 border border-gray-700 flex items-center gap-2">
                <FileText className="w-3 h-3" />
                {documents.find(d => d.document_id === activeDocId)?.filename}
              </span>
            ) : (
              <span className="text-gray-500 italic">None selected</span>
            )}
          </div>
          <button 
            onClick={clearChat}
            disabled={messages.length === 0}
            className="text-sm px-3 py-1.5 rounded hover:bg-gray-800 text-gray-400 disabled:opacity-50 transition-colors"
          >
            Clear Chat
          </button>
        </div>

        <div className="flex-1 overflow-y-auto p-6 space-y-6">
          {messages.length === 0 && (
            <div className="h-full flex flex-col items-center justify-center text-gray-500 space-y-4">
              <RefreshCw className="w-12 h-12 opacity-20" />
              <p>Ask a question about the active document.</p>
            </div>
          )}
          
          {messages.map((msg, i) => (
            <div key={i} className={`flex ${msg.role === 'user' ? 'justify-end' : 'justify-start'}`}>
              <div className={`max-w-[80%] rounded-2xl px-5 py-4 shadow-sm ${
                msg.role === 'user' 
                  ? 'bg-orange-600 text-white rounded-tr-sm' 
                  : 'bg-gray-800 text-gray-200 rounded-tl-sm border border-gray-700'
              }`}>
                {msg.role === 'assistant' ? (
                  <div className="prose prose-invert prose-orange max-w-none prose-p:leading-relaxed prose-pre:bg-gray-900 prose-pre:border prose-pre:border-gray-700">
                    <ReactMarkdown>{msg.content}</ReactMarkdown>
                  </div>
                ) : (
                  <div className="whitespace-pre-wrap">{msg.content}</div>
                )}
                
                {msg.sourceDocument && (
                  <div className="mt-4 pt-3 border-t border-gray-700 flex items-center gap-1.5 text-xs text-gray-400">
                    <FileText className="w-3 h-3" />
                    Source: <span className="text-gray-300 font-medium">{msg.sourceDocument}</span>
                  </div>
                )}
              </div>
            </div>
          ))}
          {isAnswering && (
            <div className="flex justify-start">
              <div className="bg-gray-800 border border-gray-700 rounded-2xl rounded-tl-sm px-5 py-4 flex items-center gap-2 text-gray-400">
                <Loader2 className="w-4 h-4 animate-spin" />
                Thinking...
              </div>
            </div>
          )}
          <div ref={messagesEndRef} />
        </div>

        <div className="p-6 bg-gray-900 border-t border-gray-800">
          <div className="max-w-4xl mx-auto relative flex items-end bg-gray-800 rounded-xl border border-gray-700 focus-within:border-orange-500/50 focus-within:ring-1 focus-within:ring-orange-500/50 transition-all shadow-sm">
            <textarea
              value={input}
              onChange={(e) => setInput(e.target.value)}
              onKeyDown={handleKeyDown}
              disabled={!activeDocId || isAnswering}
              placeholder={activeDocId ? "Ask anything... (Shift+Enter for new line)" : "Select a document to ask questions..."}
              className="w-full bg-transparent text-gray-100 placeholder-gray-500 resize-none max-h-48 min-h-[56px] p-4 outline-none disabled:opacity-50"
              rows={1}
              style={{ height: 'auto', minHeight: '56px' }}
            />
            <div className="p-3">
              <button
                onClick={handleSend}
                disabled={!input.trim() || !activeDocId || isAnswering}
                className="p-2 bg-orange-600 hover:bg-orange-500 disabled:bg-gray-700 disabled:text-gray-500 text-white rounded-lg transition-colors flex-shrink-0"
              >
                <Send className="w-4 h-4" />
              </button>
            </div>
          </div>
        </div>
      </div>
    </div>
  );
}

export default App;
