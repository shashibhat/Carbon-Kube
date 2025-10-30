"""
Carbon-Kube CDK Stack

Creates EKS cluster with Carbon-Kube scheduler plugin and all supporting infrastructure.
"""

import json
from typing import Dict, List

from aws_cdk import (
    Duration,
    Stack,
    CfnOutput,
    RemovalPolicy,
    aws_ec2 as ec2,
    aws_eks as eks,
    aws_iam as iam,
    aws_s3 as s3,
    aws_lambda as lambda_,
    aws_logs as logs,
    custom_resources as cr,
)
from constructs import Construct


class CarbonKubeStack(Stack):
    """Main CDK stack for Carbon-Kube deployment."""
    
    def __init__(
        self, 
        scope: Construct, 
        construct_id: str, 
        region_type: str = "green",
        **kwargs
    ) -> None:
        super().__init__(scope, construct_id, **kwargs)
        
        self.region_type = region_type
        self.cluster_name = f"carbon-kube-{region_type}"
        
        # Create VPC
        self.vpc = self._create_vpc()
        
        # Create EKS cluster
        self.cluster = self._create_eks_cluster()
        
        # Create node groups
        self._create_node_groups()
        
        # Create IAM roles and policies
        self._create_iam_resources()
        
        # Create S3 bucket for logs and metrics
        self.metrics_bucket = self._create_s3_bucket()
        
        # Install Carbon-Kube via Helm
        self._install_carbon_kube()
        
        # Create monitoring resources
        self._create_monitoring()
        
        # Create test workloads
        self._create_test_workloads()
        
        # Create outputs
        self._create_outputs()
    
    def _create_vpc(self) -> ec2.Vpc:
        """Create VPC with public and private subnets."""
        vpc = ec2.Vpc(
            self, "CarbonKubeVPC",
            vpc_name=f"{self.cluster_name}-vpc",
            max_azs=3,
            cidr="10.0.0.0/16",
            subnet_configuration=[
                ec2.SubnetConfiguration(
                    name="PublicSubnet",
                    subnet_type=ec2.SubnetType.PUBLIC,
                    cidr_mask=24
                ),
                ec2.SubnetConfiguration(
                    name="PrivateSubnet", 
                    subnet_type=ec2.SubnetType.PRIVATE_WITH_EGRESS,
                    cidr_mask=24
                )
            ],
            enable_dns_hostnames=True,
            enable_dns_support=True
        )
        
        # Tag subnets for EKS
        for subnet in vpc.public_subnets:
            subnet.node.add_metadata("kubernetes.io/role/elb", "1")
        
        for subnet in vpc.private_subnets:
            subnet.node.add_metadata("kubernetes.io/role/internal-elb", "1")
        
        return vpc
    
    def _create_eks_cluster(self) -> eks.Cluster:
        """Create EKS cluster with Carbon-Kube configuration."""
        
        # Create cluster service role
        cluster_role = iam.Role(
            self, "ClusterRole",
            assumed_by=iam.ServicePrincipal("eks.amazonaws.com"),
            managed_policies=[
                iam.ManagedPolicy.from_aws_managed_policy_name("AmazonEKSClusterPolicy")
            ]
        )
        
        # Create cluster
        cluster = eks.Cluster(
            self, "CarbonKubeCluster",
            cluster_name=self.cluster_name,
            version=eks.KubernetesVersion.V1_28,
            vpc=self.vpc,
            vpc_subnets=[ec2.SubnetSelection(subnet_type=ec2.SubnetType.PRIVATE_WITH_EGRESS)],
            role=cluster_role,
            default_capacity=0,  # We'll add managed node groups separately
            endpoint_access=eks.EndpointAccess.PUBLIC_AND_PRIVATE,
            cluster_logging=[
                eks.ClusterLoggingTypes.API,
                eks.ClusterLoggingTypes.AUDIT,
                eks.ClusterLoggingTypes.SCHEDULER,
                eks.ClusterLoggingTypes.CONTROLLER_MANAGER
            ]
        )
        
        # Add cluster tags for carbon tracking
        cluster.node.add_metadata("carbon-kube/region-type", self.region_type)
        cluster.node.add_metadata("carbon-kube/version", "1.0.0")
        
        return cluster
    
    def _create_node_groups(self):
        """Create EKS managed node groups for different workload types."""
        
        # Node group for system workloads (always-on)
        system_node_group = self.cluster.add_nodegroup_capacity(
            "SystemNodes",
            instance_types=[ec2.InstanceType("t3.medium")],
            min_size=2,
            max_size=4,
            desired_size=2,
            subnets=ec2.SubnetSelection(subnet_type=ec2.SubnetType.PRIVATE_WITH_EGRESS),
            labels={
                "node-type": "system",
                "carbon-kube/workload-type": "system"
            },
            taints=[
                eks.TaintSpec(
                    effect=eks.TaintEffect.NO_SCHEDULE,
                    key="node-type",
                    value="system"
                )
            ]
        )
        
        # Node group for batch workloads (can be migrated)
        batch_node_group = self.cluster.add_nodegroup_capacity(
            "BatchNodes",
            instance_types=[ec2.InstanceType("m5.large"), ec2.InstanceType("m5.xlarge")],
            min_size=1,
            max_size=10,
            desired_size=2,
            subnets=ec2.SubnetSelection(subnet_type=ec2.SubnetType.PRIVATE_WITH_EGRESS),
            labels={
                "node-type": "batch",
                "carbon-kube/workload-type": "batch",
                "carbon-kube/migratable": "true"
            },
            capacity_type=eks.CapacityType.SPOT,  # Use spot instances for cost efficiency
        )
        
        # Node group for ML workloads (GPU-enabled)
        if self.region_type == "green":  # Only in green regions for cost optimization
            ml_node_group = self.cluster.add_nodegroup_capacity(
                "MLNodes",
                instance_types=[ec2.InstanceType("g4dn.xlarge")],
                min_size=0,
                max_size=5,
                desired_size=0,  # Scale from zero
                subnets=ec2.SubnetSelection(subnet_type=ec2.SubnetType.PRIVATE_WITH_EGRESS),
                labels={
                    "node-type": "ml",
                    "carbon-kube/workload-type": "ml",
                    "carbon-kube/migratable": "true"
                }
            )
    
    def _create_iam_resources(self):
        """Create IAM roles and policies for Carbon-Kube components."""
        
        # Policy for accessing carbon intensity APIs
        carbon_api_policy = iam.PolicyDocument(
            statements=[
                iam.PolicyStatement(
                    effect=iam.Effect.ALLOW,
                    actions=[
                        "secretsmanager:GetSecretValue"
                    ],
                    resources=[
                        f"arn:aws:secretsmanager:{self.region}:{self.account}:secret:carbon-kube/*"
                    ]
                ),
                iam.PolicyStatement(
                    effect=iam.Effect.ALLOW,
                    actions=[
                        "s3:GetObject",
                        "s3:PutObject",
                        "s3:DeleteObject"
                    ],
                    resources=[
                        f"{self.metrics_bucket.bucket_arn}/*"
                    ]
                ),
                iam.PolicyStatement(
                    effect=iam.Effect.ALLOW,
                    actions=[
                        "cloudwatch:PutMetricData",
                        "cloudwatch:GetMetricStatistics"
                    ],
                    resources=["*"]
                )
            ]
        )
        
        # Service account for Carbon-Kube poller
        poller_service_account = self.cluster.add_service_account(
            "CarbonPollerServiceAccount",
            name="carbon-poller",
            namespace="default"
        )
        
        poller_service_account.role.attach_inline_policy(
            iam.Policy(
                self, "CarbonPollerPolicy",
                document=carbon_api_policy
            )
        )
        
        # Service account for RL tuner
        rl_service_account = self.cluster.add_service_account(
            "RLTunerServiceAccount", 
            name="rl-tuner",
            namespace="default"
        )
        
        rl_service_account.role.attach_inline_policy(
            iam.Policy(
                self, "RLTunerPolicy",
                document=carbon_api_policy
            )
        )
    
    def _create_s3_bucket(self) -> s3.Bucket:
        """Create S3 bucket for storing metrics and logs."""
        bucket = s3.Bucket(
            self, "MetricsBucket",
            bucket_name=f"carbon-kube-metrics-{self.account}-{self.region}",
            versioned=True,
            encryption=s3.BucketEncryption.S3_MANAGED,
            removal_policy=RemovalPolicy.DESTROY,  # For demo purposes
            lifecycle_rules=[
                s3.LifecycleRule(
                    id="DeleteOldMetrics",
                    expiration=Duration.days(90),
                    noncurrent_version_expiration=Duration.days(30)
                )
            ]
        )
        
        return bucket
    
    def _install_carbon_kube(self):
        """Install Carbon-Kube using Helm chart."""
        
        # Create namespace
        namespace = self.cluster.add_manifest("CarbonKubeNamespace", {
            "apiVersion": "v1",
            "kind": "Namespace", 
            "metadata": {
                "name": "carbon-kube",
                "labels": {
                    "name": "carbon-kube"
                }
            }
        })
        
        # Install Katalyst (prerequisite)
        katalyst_chart = self.cluster.add_helm_chart(
            "Katalyst",
            chart="katalyst",
            repository="https://kubewharf.github.io/charts",
            namespace="katalyst-system",
            create_namespace=True,
            values={
                "scheduler": {
                    "enabled": True,
                    "plugins": {
                        "enabled": ["EmissionPlugin"]
                    }
                }
            }
        )
        
        # Install Carbon-Kube chart
        carbon_kube_chart = self.cluster.add_helm_chart(
            "CarbonKube",
            chart="./charts/carbon-kube",  # Local chart
            namespace="carbon-kube",
            values={
                "image": {
                    "repository": "carbon-kube/carbon-kube",
                    "tag": "v1.0.0"
                },
                "poller": {
                    "enabled": True,
                    "schedule": "*/5 * * * *",  # Every 5 minutes
                    "serviceAccount": {
                        "name": "carbon-poller"
                    }
                },
                "rlTuner": {
                    "enabled": True,
                    "serviceAccount": {
                        "name": "rl-tuner"
                    }
                },
                "config": {
                    "threshold": 200.0,
                    "regions": [self.region],
                    "regionType": self.region_type
                },
                "monitoring": {
                    "prometheus": {
                        "enabled": True
                    },
                    "grafana": {
                        "enabled": True
                    }
                }
            }
        )
        
        carbon_kube_chart.node.add_dependency(katalyst_chart)
        carbon_kube_chart.node.add_dependency(namespace)
    
    def _create_monitoring(self):
        """Create monitoring and observability resources."""
        
        # Install Prometheus
        prometheus_chart = self.cluster.add_helm_chart(
            "Prometheus",
            chart="kube-prometheus-stack",
            repository="https://prometheus-community.github.io/helm-charts",
            namespace="monitoring",
            create_namespace=True,
            values={
                "prometheus": {
                    "prometheusSpec": {
                        "retention": "30d",
                        "storageSpec": {
                            "volumeClaimTemplate": {
                                "spec": {
                                    "storageClassName": "gp2",
                                    "accessModes": ["ReadWriteOnce"],
                                    "resources": {
                                        "requests": {
                                            "storage": "50Gi"
                                        }
                                    }
                                }
                            }
                        }
                    }
                },
                "grafana": {
                    "enabled": True,
                    "adminPassword": "carbon-kube-admin",  # Change in production
                    "service": {
                        "type": "LoadBalancer"
                    }
                }
            }
        )
        
        # Create CloudWatch log group for EKS logs
        log_group = logs.LogGroup(
            self, "EKSLogGroup",
            log_group_name=f"/aws/eks/{self.cluster_name}/cluster",
            retention=logs.RetentionDays.ONE_MONTH,
            removal_policy=RemovalPolicy.DESTROY
        )
    
    def _create_test_workloads(self):
        """Create test workloads for evaluating Carbon-Kube."""
        
        # Spark job for big data processing
        spark_job = self.cluster.add_manifest("SparkTestJob", {
            "apiVersion": "batch/v1",
            "kind": "Job",
            "metadata": {
                "name": "carbon-test-spark",
                "namespace": "default",
                "labels": {
                    "carbon-kube/workload-type": "batch",
                    "carbon-kube/migratable": "true"
                }
            },
            "spec": {
                "template": {
                    "spec": {
                        "containers": [{
                            "name": "spark-pi",
                            "image": "apache/spark:3.4.0",
                            "command": [
                                "/opt/spark/bin/spark-submit",
                                "--class", "org.apache.spark.examples.SparkPi",
                                "--master", "local[4]",
                                "/opt/spark/examples/jars/spark-examples_2.12-3.4.0.jar",
                                "1000"
                            ],
                            "resources": {
                                "requests": {
                                    "cpu": "2000m",
                                    "memory": "4Gi"
                                },
                                "limits": {
                                    "cpu": "4000m", 
                                    "memory": "8Gi"
                                }
                            }
                        }],
                        "restartPolicy": "Never",
                        "nodeSelector": {
                            "node-type": "batch"
                        },
                        "tolerations": [{
                            "key": "carbon-kube/migratable",
                            "operator": "Equal",
                            "value": "true",
                            "effect": "NoSchedule"
                        }]
                    }
                },
                "backoffLimit": 3
            }
        })
        
        # Flink job for stream processing
        flink_job = self.cluster.add_manifest("FlinkTestJob", {
            "apiVersion": "apps/v1",
            "kind": "Deployment",
            "metadata": {
                "name": "carbon-test-flink",
                "namespace": "default",
                "labels": {
                    "carbon-kube/workload-type": "streaming",
                    "carbon-kube/migratable": "true"
                }
            },
            "spec": {
                "replicas": 2,
                "selector": {
                    "matchLabels": {
                        "app": "carbon-test-flink"
                    }
                },
                "template": {
                    "metadata": {
                        "labels": {
                            "app": "carbon-test-flink",
                            "carbon-kube/workload-type": "streaming"
                        }
                    },
                    "spec": {
                        "containers": [{
                            "name": "flink-taskmanager",
                            "image": "flink:1.17",
                            "command": ["taskmanager.sh", "start-foreground"],
                            "resources": {
                                "requests": {
                                    "cpu": "1000m",
                                    "memory": "2Gi"
                                },
                                "limits": {
                                    "cpu": "2000m",
                                    "memory": "4Gi"
                                }
                            }
                        }],
                        "nodeSelector": {
                            "node-type": "batch"
                        }
                    }
                }
            }
        })
    
    def _create_outputs(self):
        """Create CloudFormation outputs."""
        
        CfnOutput(
            self, "ClusterName",
            value=self.cluster.cluster_name,
            description="EKS Cluster Name"
        )
        
        CfnOutput(
            self, "ClusterEndpoint", 
            value=self.cluster.cluster_endpoint,
            description="EKS Cluster Endpoint"
        )
        
        CfnOutput(
            self, "KubeconfigCommand",
            value=f"aws eks update-kubeconfig --region {self.region} --name {self.cluster.cluster_name}",
            description="Command to update kubeconfig"
        )
        
        CfnOutput(
            self, "MetricsBucket",
            value=self.metrics_bucket.bucket_name,
            description="S3 bucket for metrics storage"
        )
        
        CfnOutput(
            self, "RegionType",
            value=self.region_type,
            description="Region type (green/mixed/dirty) for carbon intensity"
        )