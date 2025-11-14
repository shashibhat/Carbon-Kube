#!/usr/bin/env python3
import aws_cdk as cdk

from carbon_kube_stack import CarbonKubeStack


app = cdk.App()
CarbonKubeStack(app, "CarbonKubeStack")
app.synth()
