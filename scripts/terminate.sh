#!/usr/bin/env bash
set -euo pipefail

# Regions to clean up
REGIONS=("us-west-1" "us-west-2")

for REGION in "${REGIONS[@]}"; do
  echo "=================================================="
  echo " Shutting down EC2 + EKS in region: ${REGION}"
  echo "=================================================="

  ########################################
  # EC2 INSTANCES
  ########################################
  echo "[${REGION}] EC2: Stopping all running instances..."
  EC2_IDS=$(aws ec2 describe-instances \
    --region "${REGION}" \
    --filters "Name=instance-state-name,Values=running" \
    --query "Reservations[*].Instances[*].InstanceId" \
    --output text || true)

  if [[ -n "${EC2_IDS:-}" ]]; then
    echo "[${REGION}] EC2: Found instances: ${EC2_IDS}"
    aws ec2 stop-instances --region "${REGION}" --instance-ids ${EC2_IDS}
  else
    echo "[${REGION}] EC2: No running instances."
  fi

  ########################################
  # AUTOSCALING GROUPS
  ########################################
  echo "[${REGION}] ASG: Scaling all Auto Scaling Groups to 0..."
  ASG_NAMES=$(aws autoscaling describe-auto-scaling-groups \
    --region "${REGION}" \
    --query "AutoScalingGroups[*].AutoScalingGroupName" \
    --output text || true)

  if [[ -n "${ASG_NAMES:-}" ]]; then
    for ASG in ${ASG_NAMES}; do
      echo "[${REGION}] ASG: Scaling ${ASG} -> min=0, desired=0, max=0"
      aws autoscaling update-auto-scaling-group \
        --region "${REGION}" \
        --auto-scaling-group-name "${ASG}" \
        --min-size 0 \
        --desired-capacity 0 \
        --max-size 0
    done
  else
    echo "[${REGION}] ASG: No Auto Scaling Groups found."
  fi

  ########################################
  # EKS MANAGED NODEGROUPS
  ########################################
  echo "[${REGION}] EKS: Scaling all managed nodegroups to 0..."
  EKS_CLUSTERS=$(aws eks list-clusters \
    --region "${REGION}" \
    --query "clusters" \
    --output text || true)

  if [[ -n "${EKS_CLUSTERS:-}" ]]; then
    for CLUSTER in ${EKS_CLUSTERS}; do
      echo "[${REGION}] EKS: Processing cluster ${CLUSTER}"
      NODEGROUPS=$(aws eks list-nodegroups \
        --region "${REGION}" \
        --cluster-name "${CLUSTER}" \
        --query "nodegroups" \
        --output text || true)

      if [[ -n "${NODEGROUPS:-}" ]]; then
        for NG in ${NODEGROUPS}; do
          echo "[${REGION}] EKS: Scaling nodegroup ${NG} -> min=0, desired=0, max=0"
          aws eks update-nodegroup-config \
            --region "${REGION}" \
            --cluster-name "${CLUSTER}" \
            --nodegroup-name "${NG}" \
            --scaling-config minSize=0,maxSize=0,desiredSize=0
        done
      else
        echo "[${REGION}] EKS: No managed nodegroups in cluster ${CLUSTER}"
      fi
    done
  else
    echo "[${REGION}] EKS: No clusters found."
  fi

  echo "Done for region: ${REGION}"
  echo
done

echo "=================================================="
echo " Shutdown complete for EC2 + EKS in us-west-1 & us-west-2."
echo " Note: EKS control planes stay running (AWS-managed)."
echo "=================================================="
