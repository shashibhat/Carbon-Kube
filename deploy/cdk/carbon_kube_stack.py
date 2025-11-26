import aws_cdk as cdk
from aws_cdk import aws_eks as eks
from aws_cdk import aws_s3_assets as s3_assets
from aws_cdk.lambda_layer_kubectl_v28 import KubectlV28Layer
from constructs import Construct
#from aws_cdk.lambda_layer_kubectl_v28 import KubectlLayer

class CarbonKubeStack(cdk.Stack):
    def __init__(self, scope: Construct, id: str, **kwargs):
        super().__init__(scope, id, **kwargs)

        cluster = eks.Cluster(
            self,
            "Cluster",
            version=eks.KubernetesVersion.V1_28,
            default_capacity=2,
            kubectl_layer=KubectlV28Layer(self, "KubectlLayer"),
        )