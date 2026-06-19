export type ApiResponse<T> = {
  data:    T
  success: boolean
}

export type ApiError = {
  erro:     string
  detalhe?: string
}
