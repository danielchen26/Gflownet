import { useEffect, useState, useRef, useCallback } from 'react'

interface WebSocketMessage {
  type: string
  data?: any
  timestamp?: string
}

export function useWebSocket(url: string = 'ws://localhost:8080/ws') {
  const [isConnected, setIsConnected] = useState(false)
  const [lastMessage, setLastMessage] = useState<WebSocketMessage | null>(null)
  const ws = useRef<WebSocket | null>(null)
  const reconnectTimer = useRef<NodeJS.Timeout>()
  const reconnectAttempts = useRef(0)
  
  const connect = useCallback(() => {
    try {
      ws.current = new WebSocket(url)
      
      ws.current.onopen = () => {
        console.log('WebSocket connected')
        setIsConnected(true)
        reconnectAttempts.current = 0
        
        // Subscribe to updates
        ws.current?.send(JSON.stringify({
          type: 'subscribe',
          channels: ['training', 'trajectories']
        }))
      }
      
      ws.current.onmessage = (event) => {
        try {
          const message = JSON.parse(event.data)
          setLastMessage(message)
        } catch (error) {
          console.error('Failed to parse WebSocket message:', error)
        }
      }
      
      ws.current.onclose = () => {
        console.log('WebSocket disconnected')
        setIsConnected(false)
        
        // Exponential backoff reconnection
        const delay = Math.min(1000 * Math.pow(2, reconnectAttempts.current), 30000)
        reconnectAttempts.current++
        
        reconnectTimer.current = setTimeout(() => {
          console.log(`Attempting to reconnect... (attempt ${reconnectAttempts.current})`)
          connect()
        }, delay)
      }
      
      ws.current.onerror = (error) => {
        console.error('WebSocket error:', error)
      }
    } catch (error) {
      console.error('Failed to create WebSocket connection:', error)
    }
  }, [url])
  
  const sendMessage = useCallback((message: WebSocketMessage) => {
    if (ws.current?.readyState === WebSocket.OPEN) {
      ws.current.send(JSON.stringify(message))
    }
  }, [])
  
  useEffect(() => {
    connect()
    
    // Ping every 30 seconds to keep connection alive
    const pingInterval = setInterval(() => {
      sendMessage({ type: 'ping' })
    }, 30000)
    
    return () => {
      clearInterval(pingInterval)
      clearTimeout(reconnectTimer.current)
      ws.current?.close()
    }
  }, [connect, sendMessage])
  
  return {
    isConnected,
    lastMessage,
    sendMessage
  }
}