import aws_cdk as cdk
from aws_cdk import aws_eks as eks
from constructs import Construct


class CarbonKubeStack(cdk.Stack):
    """Minimal EKS + Helm stack for Carbon-Kube."""

    def __init__(self, scope: Construct, construct_id: str, **kwargs) -> None:
        super().__init__(scope, construct_id, **kwargs)

        cluster = eks.Cluster(
            self,
            "Cluster",
            version=eks.KubernetesVersion.V1_28,
            default_capacity=2,
        )

        cluster.add_helm_chart(
            "CarbonKubeChart",
            chart="./deploy/helm",
            release="carbon-kube",
        )
