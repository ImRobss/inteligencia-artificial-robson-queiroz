export type PacienteDto = {
  gestacoes:          number
  glicose:            number
  pressao_arterial:   number
  espessura_pele:     number
  insulina:           number
  imc:                number
  historico_familiar: number
  idade:              number
}

export type RespostaPreditiva = {
  diagnostico:            string
  codigo:                 0 | 1
  probabilidade_diabetes: number
  risco:                  'Baixo' | 'Médio' | 'Alto'
}
