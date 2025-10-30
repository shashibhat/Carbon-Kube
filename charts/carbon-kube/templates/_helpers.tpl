{{/*
Expand the name of the chart.
*/}}
{{- define "carbon-kube.name" -}}
{{- default .Chart.Name .Values.nameOverride | trunc 63 | trimSuffix "-" }}
{{- end }}

{{/*
Create a default fully qualified app name.
We truncate at 63 chars because some Kubernetes name fields are limited to this (by the DNS naming spec).
If release name contains chart name it will be used as a full name.
*/}}
{{- define "carbon-kube.fullname" -}}
{{- if .Values.fullnameOverride }}
{{- .Values.fullnameOverride | trunc 63 | trimSuffix "-" }}
{{- else }}
{{- $name := default .Chart.Name .Values.nameOverride }}
{{- if contains $name .Release.Name }}
{{- .Release.Name | trunc 63 | trimSuffix "-" }}
{{- else }}
{{- printf "%s-%s" .Release.Name $name | trunc 63 | trimSuffix "-" }}
{{- end }}
{{- end }}
{{- end }}

{{/*
Create chart name and version as used by the chart label.
*/}}
{{- define "carbon-kube.chart" -}}
{{- printf "%s-%s" .Chart.Name .Chart.Version | replace "+" "_" | trunc 63 | trimSuffix "-" }}
{{- end }}

{{/*
Common labels
*/}}
{{- define "carbon-kube.labels" -}}
helm.sh/chart: {{ include "carbon-kube.chart" . }}
{{ include "carbon-kube.selectorLabels" . }}
{{- if .Chart.AppVersion }}
app.kubernetes.io/version: {{ .Chart.AppVersion | quote }}
{{- end }}
app.kubernetes.io/managed-by: {{ .Release.Service }}
carbon-kube/version: {{ .Chart.AppVersion | quote }}
{{- end }}

{{/*
Selector labels
*/}}
{{- define "carbon-kube.selectorLabels" -}}
app.kubernetes.io/name: {{ include "carbon-kube.name" . }}
app.kubernetes.io/instance: {{ .Release.Name }}
{{- end }}

{{/*
Create the name of the service account to use
*/}}
{{- define "carbon-kube.serviceAccountName" -}}
{{- if .Values.serviceAccount.create }}
{{- default (include "carbon-kube.fullname" .) .Values.serviceAccount.name }}
{{- else }}
{{- default "default" .Values.serviceAccount.name }}
{{- end }}
{{- end }}

{{/*
Create the name of the poller service account to use
*/}}
{{- define "carbon-kube.pollerServiceAccountName" -}}
{{- if .Values.poller.serviceAccount.create }}
{{- default (printf "%s-poller" (include "carbon-kube.fullname" .)) .Values.poller.serviceAccount.name }}
{{- else }}
{{- default "default" .Values.poller.serviceAccount.name }}
{{- end }}
{{- end }}

{{/*
Create the name of the RL tuner service account to use
*/}}
{{- define "carbon-kube.rlTunerServiceAccountName" -}}
{{- if .Values.rlTuner.serviceAccount.create }}
{{- default (printf "%s-rl-tuner" (include "carbon-kube.fullname" .)) .Values.rlTuner.serviceAccount.name }}
{{- else }}
{{- default "default" .Values.rlTuner.serviceAccount.name }}
{{- end }}
{{- end }}

{{/*
Create image name
*/}}
{{- define "carbon-kube.image" -}}
{{- $registry := .Values.global.imageRegistry | default .Values.image.registry -}}
{{- $repository := .Values.image.repository -}}
{{- $tag := .Values.image.tag | default .Chart.AppVersion -}}
{{- if $registry }}
{{- printf "%s/%s:%s" $registry $repository $tag }}
{{- else }}
{{- printf "%s:%s" $repository $tag }}
{{- end }}
{{- end }}

{{/*
Create scheduler image name
*/}}
{{- define "carbon-kube.schedulerImage" -}}
{{- $registry := .Values.global.imageRegistry | default .Values.image.registry -}}
{{- $repository := .Values.scheduler.image.repository -}}
{{- $tag := .Values.scheduler.image.tag | default .Chart.AppVersion -}}
{{- if $registry }}
{{- printf "%s/%s:%s" $registry $repository $tag }}
{{- else }}
{{- printf "%s:%s" $repository $tag }}
{{- end }}
{{- end }}

{{/*
Create poller image name
*/}}
{{- define "carbon-kube.pollerImage" -}}
{{- $registry := .Values.global.imageRegistry | default .Values.image.registry -}}
{{- $repository := .Values.poller.image.repository -}}
{{- $tag := .Values.poller.image.tag | default .Chart.AppVersion -}}
{{- if $registry }}
{{- printf "%s/%s:%s" $registry $repository $tag }}
{{- else }}
{{- printf "%s:%s" $repository $tag }}
{{- end }}
{{- end }}

{{/*
Create RL tuner image name
*/}}
{{- define "carbon-kube.rlTunerImage" -}}
{{- $registry := .Values.global.imageRegistry | default .Values.image.registry -}}
{{- $repository := .Values.rlTuner.image.repository -}}
{{- $tag := .Values.rlTuner.image.tag | default .Chart.AppVersion -}}
{{- if $registry }}
{{- printf "%s/%s:%s" $registry $repository $tag }}
{{- else }}
{{- printf "%s:%s" $repository $tag }}
{{- end }}
{{- end }}

{{/*
Create ConfigMap name
*/}}
{{- define "carbon-kube.configMapName" -}}
{{- default (printf "%s-config" (include "carbon-kube.fullname" .)) .Values.configMap.name }}
{{- end }}

{{/*
Create Secret name
*/}}
{{- define "carbon-kube.secretName" -}}
{{- default (printf "%s-secrets" (include "carbon-kube.fullname" .)) .Values.secret.name }}
{{- end }}

{{/*
Common environment variables
*/}}
{{- define "carbon-kube.commonEnvVars" -}}
- name: CARBON_KUBE_NAMESPACE
  valueFrom:
    fieldRef:
      fieldPath: metadata.namespace
- name: CARBON_KUBE_POD_NAME
  valueFrom:
    fieldRef:
      fieldPath: metadata.name
- name: CARBON_KUBE_NODE_NAME
  valueFrom:
    fieldRef:
      fieldPath: spec.nodeName
- name: CARBON_KUBE_CONFIG_MAP
  value: {{ include "carbon-kube.configMapName" . }}
- name: CARBON_KUBE_SECRET
  value: {{ include "carbon-kube.secretName" . }}
- name: CARBON_KUBE_THRESHOLD
  value: {{ .Values.config.threshold | quote }}
- name: CARBON_KUBE_REGIONS
  value: {{ join "," .Values.config.regions | quote }}
- name: CARBON_KUBE_REGION_TYPE
  value: {{ .Values.config.regionType | quote }}
- name: CARBON_KUBE_DEBUG
  value: {{ .Values.config.debug | quote }}
{{- if .Values.extraEnvVars }}
{{ toYaml .Values.extraEnvVars }}
{{- end }}
{{- end }}

{{/*
Common volume mounts
*/}}
{{- define "carbon-kube.commonVolumeMounts" -}}
{{- if .Values.extraVolumeMounts }}
{{ toYaml .Values.extraVolumeMounts }}
{{- end }}
{{- end }}

{{/*
Common volumes
*/}}
{{- define "carbon-kube.commonVolumes" -}}
{{- if .Values.extraVolumes }}
{{ toYaml .Values.extraVolumes }}
{{- end }}
{{- end }}

{{/*
Pod security context
*/}}
{{- define "carbon-kube.podSecurityContext" -}}
{{- if .Values.podSecurityContext }}
{{ toYaml .Values.podSecurityContext }}
{{- end }}
{{- end }}

{{/*
Container security context
*/}}
{{- define "carbon-kube.securityContext" -}}
{{- if .Values.securityContext }}
{{ toYaml .Values.securityContext }}
{{- end }}
{{- end }}

{{/*
Node selector
*/}}
{{- define "carbon-kube.nodeSelector" -}}
{{- if .Values.nodeSelector }}
{{ toYaml .Values.nodeSelector }}
{{- end }}
{{- end }}

{{/*
Tolerations
*/}}
{{- define "carbon-kube.tolerations" -}}
{{- if .Values.tolerations }}
{{ toYaml .Values.tolerations }}
{{- end }}
{{- end }}

{{/*
Affinity
*/}}
{{- define "carbon-kube.affinity" -}}
{{- if .Values.affinity }}
{{ toYaml .Values.affinity }}
{{- end }}
{{- end }}