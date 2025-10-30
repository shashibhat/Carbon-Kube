#!/usr/bin/env python3
"""
AWS CDK App for Carbon-Kube Deployment

This CDK app creates a complete EKS cluster with Carbon-Kube installed,
supporting multi-region deployment for carbon-aware workload scheduling.
"""

import os
from aws_cdk import App, Environment
from constructs import Construct

from stacks.carbon_kube_stack import CarbonKubeStack


def main():
    """Main CDK app entry point."""
    app = App()
    
    # Get deployment configuration from environment
    account = os.getenv('CDK_DEFAULT_ACCOUNT', app.node.try_get_context('account'))
    region = os.getenv('CDK_DEFAULT_REGION', app.node.try_get_context('region') or 'us-west-2')
    
    # Environment configuration
    env = Environment(account=account, region=region)
    
    # Deploy primary stack
    primary_stack = CarbonKubeStack(
        app, 
        "CarbonKubePrimary",
        env=env,
        region_type="green",  # Primary region should be green (renewable-heavy)
        description="Carbon-Kube primary deployment in green region"
    )
    
    # Deploy secondary stack in a different region for testing
    secondary_region = 'us-east-1' if region == 'us-west-2' else 'us-west-2'
    secondary_env = Environment(account=account, region=secondary_region)
    
    secondary_stack = CarbonKubeStack(
        app,
        "CarbonKubeSecondary", 
        env=secondary_env,
        region_type="mixed",  # Secondary region for comparison
        description="Carbon-Kube secondary deployment for multi-region testing"
    )
    
    # Add cross-stack dependencies if needed
    secondary_stack.add_dependency(primary_stack)
    
    app.synth()


if __name__ == "__main__":
    main()