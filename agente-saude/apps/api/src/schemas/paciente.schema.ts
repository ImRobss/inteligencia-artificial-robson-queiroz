/**
 * paciente.schema.ts
 * JSON Schema para validação automática do Fastify +
 * tipos TypeScript derivados para uso no controller/service.
 */

export const pacienteSchema = {
  type: 'object',
  required: [
    'gestacoes','glicose','pressao_arterial','espessura_pele',
    'insulina','imc','historico_familiar','idade',
  ],
  additionalProperties: false,
  properties: {
    gestacoes:          { type: 'number', minimum: 0,   maximum: 20,   description: 'Número de gestações' },
    glicose:            { type: 'number', minimum: 1,   maximum: 500,  description: 'Glicose plasmática (mg/dL)' },
    pressao_arterial:   { type: 'number', minimum: 0,   maximum: 200,  description: 'Pressão arterial diastólica (mmHg)' },
    espessura_pele:     { type: 'number', minimum: 0,   maximum: 100,  description: 'Dobra cutânea (mm)' },
    insulina:           { type: 'number', minimum: 0,   maximum: 1000, description: 'Insulina sérica (mu U/ml)' },
    imc:                { type: 'number', minimum: 1,   maximum: 100,  description: 'IMC (kg/m²)' },
    historico_familiar: { type: 'number', minimum: 0,   maximum: 2.5,  description: 'Função histórico familiar' },
    idade:              { type: 'number', minimum: 1,   maximum: 120,  description: 'Idade em anos' },
  },
} as const

export const respostaSchema = {
  type: 'object',
  properties: {
    diagnostico:            { type: 'string' },
    codigo:                 { type: 'integer' },
    probabilidade_diabetes: { type: 'number' },
    risco:                  { type: 'string' },
  },
} as const

// Tipos TypeScript inferidos do schema
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
  codigo:                 number
  probabilidade_diabetes: number
  risco:                  string
}
