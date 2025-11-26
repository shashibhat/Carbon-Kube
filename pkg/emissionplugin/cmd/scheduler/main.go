package main

import (
	"k8s.io/apimachinery/pkg/runtime"
	"k8s.io/kubernetes/pkg/scheduler/framework"

	"github.com/carbon-kube/carbon-kube/pkg/emissionplugin"
)

// PluginFactory is the factory function for creating the emission plugin
// This is used by the Kubernetes scheduler framework to instantiate the plugin
func PluginFactory(args runtime.Object, handle framework.Handle) (framework.Plugin, error) {
	return emissionplugin.NewEmissionPlugin(args, handle)
}

// PluginName returns the name of the plugin
func PluginName() string {
	return emissionplugin.PluginName
}

// This file serves as the entry point for the scheduler plugin
// The plugin is built as a shared library (.so) and loaded dynamically
// by the Kubernetes scheduler framework


