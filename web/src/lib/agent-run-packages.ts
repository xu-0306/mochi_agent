import type * as api from '@/lib/api'

export const ATTEMPT_BUNDLE_MANIFEST_VERSION = 'mochi.agent_run.attempt_bundle.v1'
export const DATASET_PACKAGE_MANIFEST_VERSION = 'mochi.agent_run.dataset_package.v1'

function isRecord(value: unknown): value is Record<string, unknown> {
  return typeof value === 'object' && value !== null
}

function getString(value: unknown): string | null {
  return typeof value === 'string' && value.trim().length > 0 ? value : null
}

function getNullableString(value: unknown): string | null {
  return getString(value)
}

function getRecordArray(value: unknown): Array<Record<string, unknown>> {
  return Array.isArray(value) ? value.filter(isRecord) : []
}

function getArtifactAttemptId(artifact: api.AgentRunArtifact): string | null {
  return getString(artifact.metadata.attempt_id)
}

function getEventAttemptId(event: Record<string, unknown>): string | null {
  return getString(event.attempt_id)
}

function getScheduleAttempts(run: api.AgentRunDetail): Array<Record<string, unknown>> {
  return getRecordArray(run.schedule.recent_attempts)
}

function getScheduleAttempt(
  run: api.AgentRunDetail,
  attemptId: string | null
): Record<string, unknown> | null {
  if (!attemptId) {
    return null
  }
  return (
    getScheduleAttempts(run).find((attempt) => getString(attempt.attempt_id) === attemptId) ?? null
  )
}

function scopedArtifacts(run: api.AgentRunDetail, attemptId: string | null, scope: string): api.AgentRunArtifact[] {
  if (scope === 'all') {
    return run.artifacts
  }
  if (!attemptId) {
    return run.artifacts.filter((artifact) => getArtifactAttemptId(artifact) === null)
  }
  return run.artifacts.filter((artifact) => getArtifactAttemptId(artifact) === attemptId)
}

function scopedEvents(
  run: api.AgentRunDetail,
  attemptId: string | null,
  scope: string
): Array<Record<string, unknown>> {
  if (scope === 'all') {
    return run.events
  }
  if (!attemptId) {
    return run.events.filter((event) => getEventAttemptId(event) === null)
  }
  return run.events.filter((event) => getEventAttemptId(event) === attemptId)
}

function selectedCandidateId(
  run: api.AgentRunDetail,
  datasetRecords: Array<Record<string, unknown>>
): string | null {
  for (const record of datasetRecords) {
    const target = isRecord(record.target) ? record.target : null
    const candidateId = getString(target?.candidate_id)
    if (candidateId) {
      return candidateId
    }
  }
  return getString(run.summary.selected_candidate_id)
}

function finalAnswer(
  run: api.AgentRunDetail,
  datasetRecords: Array<Record<string, unknown>>
): string | null {
  for (const record of datasetRecords) {
    const target = isRecord(record.target) ? record.target : null
    const answer = getString(target?.answer)
    if (answer) {
      return answer
    }
  }
  return getString(run.summary.final_answer)
}

function artifactContent(
  artifacts: api.AgentRunArtifact[],
  artifactType: string
): Record<string, unknown> | null {
  const artifact = artifacts.find((item) => item.artifact_type === artifactType) ?? null
  return isRecord(artifact?.metadata.content) ? artifact.metadata.content : null
}

function extractCollectorShardManifests(
  artifacts: api.AgentRunArtifact[]
): Array<Record<string, unknown>> {
  return artifacts
    .filter((artifact) => artifact.artifact_type === 'collector_shard_manifest')
    .map((artifact) => {
      const content = isRecord(artifact.metadata.content) ? artifact.metadata.content : {}
      return {
        ...content,
        artifact_id: artifact.artifact_id,
        title: artifact.title,
        uri: artifact.uri,
      }
    })
}

function filterCollectorShardManifests(
  manifests: Array<Record<string, unknown>>,
  attemptId: string | null
): Array<Record<string, unknown>> {
  return manifests.filter((manifest) => getString(manifest.attempt_id) === attemptId)
}

function buildCollectorProvenanceManifest(
  records: Array<Record<string, unknown>>
): Record<string, unknown> | null {
  const manifestRecords: Array<Record<string, unknown>> = []
  records.forEach((item, index) => {
    const record = isRecord(item.record) ? item.record : item
    const metadata = isRecord(record.metadata) ? record.metadata : null
    const provenance = isRecord(metadata?.collector_provenance) ? metadata.collector_provenance : null
    if (!provenance) {
      return
    }
    manifestRecords.push({
      ...provenance,
      record_index: index + 1,
      artifact_id: getString(item.artifact_id),
      attempt_id: getString(item.attempt_id),
    })
  })
  if (manifestRecords.length === 0) {
    return null
  }

  const countByValue = (key: string) => {
    const counts = new Map<string, number>()
    for (const item of manifestRecords) {
      const value = getString(item[key])
      if (!value) {
        continue
      }
      counts.set(value, (counts.get(value) ?? 0) + 1)
    }
    return Array.from(counts.entries())
      .sort(([left], [right]) => left.localeCompare(right))
      .map(([value, count]) => ({ value, count }))
  }

  return {
    schema_version: '1.0',
    record_count: manifestRecords.length,
    records: manifestRecords,
    adapter_counts: countByValue('adapter_name'),
    policy_disposition_counts: countByValue('policy_disposition'),
    license_counts: countByValue('license'),
  }
}

function selectedModelsByRole(run: api.AgentRunDetail): Record<string, string | null> {
  const payload = isRecord(run.selected_models_roles.by_role) ? run.selected_models_roles.by_role : null
  if (!payload) {
    return {}
  }
  return Object.fromEntries(
    Object.entries(payload)
      .filter(([key, value]) => typeof key === 'string' && typeof value === 'string')
      .map(([key, value]) => [key, String(value)])
  )
}

function verificationStatus(
  verificationSummary: Record<string, unknown> | null,
  candidateId: string | null
): string | null {
  if (!verificationSummary || !candidateId) {
    return null
  }
  for (const item of getRecordArray(verificationSummary.verifications)) {
    if (getString(item.candidate_id) === candidateId) {
      return getString(item.status)
    }
  }
  return null
}

function evidenceGateStatus(record: Record<string, unknown>, candidateId: string | null): string | null {
  const evidence = isRecord(record.evidence) ? record.evidence : null
  const evaluation = isRecord(evidence?.evaluation) ? evidence.evaluation : null
  for (const score of getRecordArray(evaluation?.scores)) {
    if (candidateId && getString(score.candidate_id) !== candidateId) {
      continue
    }
    const gate = isRecord(score.evidence_gate) ? score.evidence_gate : null
    const status = getString(gate?.status)
    if (status) {
      return status
    }
  }
  return null
}

function buildGovernedDatasetRecord(
  run: api.AgentRunDetail,
  artifact: api.AgentRunArtifact,
  record: Record<string, unknown>
): Record<string, unknown> {
  const attemptId = getArtifactAttemptId(artifact)
  const candidateId = selectedCandidateId(run, [record])
  const verificationSummary = artifactContent(
    scopedArtifacts(run, attemptId, attemptId ? attemptId : 'latest'),
    'verification_summary'
  )
  const finalAnswerValue = finalAnswer(run, [record])
  const verification = verificationStatus(verificationSummary, candidateId)
  const evidenceGate = evidenceGateStatus(record, candidateId)
  const exclusionReasons: string[] = []
  if (!finalAnswerValue) {
    exclusionReasons.push('missing_final_answer')
  }
  if (verification === 'failed') {
    exclusionReasons.push('verification_failed')
  }
  if (evidenceGate === 'failed') {
    exclusionReasons.push('evidence_gate_failed')
  }
  const models = selectedModelsByRole(run)

  return {
    artifact_id: artifact.artifact_id,
    title: artifact.title,
    uri: artifact.uri,
    run_id: run.run_id,
    attempt_id: attemptId,
    protocol_id: run.protocol_id,
    selected_candidate_id: candidateId,
    verification_status: verification,
    evidence_gate_status: evidenceGate,
    teacher_model_id: models.teacher ?? null,
    student_model_id: models.student ?? null,
    judge_model_id: models.judge ?? null,
    verifier_model_id: models.verifier ?? null,
    training_ready: exclusionReasons.length === 0,
    exclusion_reasons: exclusionReasons,
    record,
  }
}

export function buildAttemptPackageFallback(
  run: api.AgentRunDetail,
  selectedScope: string,
  attemptId: string | null
): Record<string, unknown> {
  const artifacts = scopedArtifacts(run, attemptId, selectedScope)
  const events = scopedEvents(run, attemptId, selectedScope)
  const roleOutputs = events
    .filter((event) => event.type === 'role_output' && isRecord(event.payload))
    .map((event) => ({
      role_id: getString((event.payload as Record<string, unknown>).role_id) ?? 'unknown',
      candidate_id: getString((event.payload as Record<string, unknown>).candidate_id),
      model_id: getString((event.payload as Record<string, unknown>).model_id),
      round_index:
        typeof (event.payload as Record<string, unknown>).round_index === 'number'
          ? ((event.payload as Record<string, unknown>).round_index as number)
          : null,
      content: getString((event.payload as Record<string, unknown>).content) ?? '',
      timestamp: getString(event.timestamp),
    }))
  const datasetRecords = artifacts
    .filter((artifact) => artifact.artifact_type === 'dataset_record')
    .map((artifact) => artifact.metadata.record)
    .filter((record): record is Record<string, unknown> => isRecord(record))
  const collectorShardManifests = extractCollectorShardManifests(artifacts)
  const candidateId = selectedCandidateId(run, datasetRecords)
  const finalAnswerValue = finalAnswer(run, datasetRecords)

  return {
    manifest_version: ATTEMPT_BUNDLE_MANIFEST_VERSION,
    package_type: 'attempt_bundle',
    exported_at: new Date().toISOString(),
    run_id: run.run_id,
    protocol_id: run.protocol_id,
    attempt_id: attemptId,
    selected_scope: selectedScope,
    schedule_attempt: getScheduleAttempt(run, attemptId),
    artifact_count: artifacts.length,
    event_count: events.length,
    role_output_count: roleOutputs.length,
    replay_ready: events.length > 0 && (candidateId !== null || finalAnswerValue !== null),
    artifacts: artifacts.map((artifact) => ({
      artifact_id: artifact.artifact_id,
      artifact_type: artifact.artifact_type,
      title: artifact.title,
      uri: artifact.uri,
      mime_type: artifact.mime_type,
      size_bytes: artifact.size_bytes,
      metadata: artifact.metadata,
    })),
    events,
    role_outputs: roleOutputs,
    evaluation_events: events.filter((event) => event.type === 'evaluation'),
    collector_shard_manifests: collectorShardManifests,
    collector_provenance_manifest: buildCollectorProvenanceManifest(datasetRecords),
    dataset_records: datasetRecords,
    run_summary: {
      ...run.summary,
      attempt_id: attemptId,
      artifact_content: artifactContent(artifacts, 'run_summary'),
    },
    evidence_summary: artifactContent(artifacts, 'evidence_summary'),
    verification_summary: artifactContent(artifacts, 'verification_summary'),
    final_selected_candidate:
      candidateId !== null || finalAnswerValue !== null
        ? {
            candidate_id: candidateId,
            final_answer: finalAnswerValue,
          }
        : null,
  }
}

export function buildDatasetPackageFallback(run: api.AgentRunDetail): Record<string, unknown> {
  const datasetArtifacts = run.artifacts.filter((artifact) => artifact.artifact_type === 'dataset_record')
  const collectorShardManifests = extractCollectorShardManifests(run.artifacts)
  const grouped = new Map<string, Array<Record<string, unknown>>>()
  const scheduleAttempts = getScheduleAttempts(run)

  for (const artifact of datasetArtifacts) {
    const record = isRecord(artifact.metadata.record) ? artifact.metadata.record : null
    if (!record) {
      continue
    }
    const attemptId = getArtifactAttemptId(artifact) ?? 'unscoped'
    const current = grouped.get(attemptId) ?? []
    current.push(buildGovernedDatasetRecord(run, artifact, record))
    grouped.set(attemptId, current)
  }

  const attempts: Array<Record<string, unknown>> = []
  const allRecords: Array<Record<string, unknown>> = []
  const trainingReadyRecords: Array<Record<string, unknown>> = []
  const excludedReasons = new Map<string, number>()

  const orderedAttemptIds = scheduleAttempts
    .map((attempt) => getString(attempt.attempt_id))
    .filter((attemptId): attemptId is string => Boolean(attemptId))

  for (const attemptId of grouped.keys()) {
    if (attemptId !== 'unscoped' && !orderedAttemptIds.includes(attemptId)) {
      orderedAttemptIds.push(attemptId)
    }
  }
  if (grouped.has('unscoped')) {
    orderedAttemptIds.push('unscoped')
  }

  for (const attemptId of orderedAttemptIds) {
    const records = grouped.get(attemptId) ?? []
    if (records.length === 0) {
      continue
    }
    const trainingReady = records.filter((record) => Boolean(record.training_ready))
    const excluded = records.filter((record) => !record.training_ready)
    for (const record of excluded) {
      const reasons = Array.isArray(record.exclusion_reasons) ? record.exclusion_reasons : []
      for (const reason of reasons) {
        if (typeof reason !== 'string' || reason.length === 0) {
          continue
        }
        excludedReasons.set(reason, (excludedReasons.get(reason) ?? 0) + 1)
      }
    }
    attempts.push({
      attempt_id: attemptId === 'unscoped' ? null : attemptId,
      schedule_attempt:
        attemptId === 'unscoped'
          ? null
          : scheduleAttempts.find((attempt) => getString(attempt.attempt_id) === attemptId) ?? null,
      dataset_record_count: records.length,
      training_ready_count: trainingReady.length,
      excluded_record_count: excluded.length,
      collector_shard_manifests: extractCollectorShardManifests(
        scopedArtifacts(run, attemptId === 'unscoped' ? null : attemptId, attemptId)
      ),
      collector_provenance_manifest: buildCollectorProvenanceManifest(records),
      dataset_records: records,
    })
    allRecords.push(...records)
    trainingReadyRecords.push(...trainingReady)
  }

  return {
    manifest_version: DATASET_PACKAGE_MANIFEST_VERSION,
    package_type: 'dataset_package',
    exported_at: new Date().toISOString(),
    run_id: run.run_id,
    protocol_id: run.protocol_id,
    attempt_count: attempts.length,
    dataset_record_count: allRecords.length,
    training_ready_count: trainingReadyRecords.length,
    excluded_record_count: allRecords.length - trainingReadyRecords.length,
    collector_shard_manifests: collectorShardManifests,
    collector_provenance_manifest: buildCollectorProvenanceManifest(allRecords),
    attempts,
    all_records: allRecords,
    training_ready_records: trainingReadyRecords,
    excluded_records_summary: {
      excluded_count: allRecords.length - trainingReadyRecords.length,
      reasons: Array.from(excludedReasons.entries()).map(([reason, count]) => ({ reason, count })),
    },
  }
}

export function buildTrainingReadyOnlyDatasetPackage(
  datasetPackage: Record<string, unknown>
): Record<string, unknown> {
  const allCollectorShardManifests = getRecordArray(datasetPackage.collector_shard_manifests)
  const attempts: Array<Record<string, unknown>> = getRecordArray(datasetPackage.attempts).map(
    (attempt) => {
      const datasetRecords = getRecordArray(attempt.dataset_records).filter(
        (record) => record.training_ready === true
      )
      const attemptId = getNullableString(attempt.attempt_id)
      return {
      ...attempt,
      collector_shard_manifests: filterCollectorShardManifests(
        allCollectorShardManifests,
        attemptId
      ),
      collector_provenance_manifest: buildCollectorProvenanceManifest(datasetRecords),
      dataset_record_count: datasetRecords.length,
      training_ready_count: datasetRecords.length,
        excluded_record_count: 0,
        dataset_records: datasetRecords,
      }
    }
  )
  const filteredAttempts = attempts.filter(
    (attempt) => typeof attempt.dataset_record_count === 'number' && attempt.dataset_record_count > 0
  )
  const trainingReadyRecords = getRecordArray(datasetPackage.training_ready_records)
  const allowedAttemptIds = new Set(
    filteredAttempts.map((attempt) => getNullableString(attempt.attempt_id))
  )
  return {
    ...datasetPackage,
    dataset_record_count: trainingReadyRecords.length,
    training_ready_count: trainingReadyRecords.length,
    excluded_record_count: 0,
    collector_shard_manifests: allCollectorShardManifests.filter((manifest) =>
      allowedAttemptIds.has(getNullableString(manifest.attempt_id))
    ),
    collector_provenance_manifest: buildCollectorProvenanceManifest(trainingReadyRecords),
    attempts: filteredAttempts,
    all_records: trainingReadyRecords,
    training_ready_records: trainingReadyRecords,
    excluded_records_summary: {
      excluded_count: 0,
      reasons: [],
    },
  }
}
