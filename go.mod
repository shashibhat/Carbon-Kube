module github.com/carbon-kube/carbon-kube

go 1.21

require (
	// Core Dependencies
	github.com/go-redis/redis/v8 v8.11.5
	github.com/prometheus/client_golang v1.16.0
	github.com/stretchr/testify v1.8.4
	go.uber.org/zap v1.25.0
	k8s.io/klog/v2 v2.100.1
	k8s.io/kubernetes v1.24.16
)

replace github.com/kubewharf/katalyst-core => ./third_party/kubewharf/katalyst-core

replace github.com/kubewharf/katalyst-api => ./third_party/kubewharf/katalyst-api

replace k8s.io/api => k8s.io/api v0.24.6

replace k8s.io/apimachinery => k8s.io/apimachinery v0.24.6

replace k8s.io/client-go => k8s.io/client-go v0.24.6

replace k8s.io/kubernetes => k8s.io/kubernetes v1.24.16

replace k8s.io/component-helpers => k8s.io/component-helpers v0.24.16

replace k8s.io/kube-scheduler => k8s.io/kube-scheduler v0.24.6

replace k8s.io/dynamic-resource-allocation => k8s.io/dynamic-resource-allocation v0.24.16

replace k8s.io/controller-manager => k8s.io/controller-manager v0.24.16

replace k8s.io/apiserver => k8s.io/apiserver v0.24.16

replace github.com/google/gnostic => github.com/google/gnostic v0.6.8

replace github.com/google/gnostic-models => github.com/google/gnostic-models v0.6.8

require (
	// Indirect dependencies
	github.com/beorn7/perks v1.0.1 // indirect
	github.com/cespare/xxhash/v2 v2.2.0 // indirect
	github.com/davecgh/go-spew v1.1.1 // indirect
	github.com/emicklei/go-restful/v3 v3.9.0 // indirect
	github.com/go-logr/logr v1.2.4 // indirect
	github.com/go-openapi/jsonpointer v0.19.6 // indirect
	github.com/go-openapi/jsonreference v0.20.2 // indirect
	github.com/go-openapi/swag v0.22.3 // indirect
	github.com/gogo/protobuf v1.3.2 // indirect
	github.com/golang/protobuf v1.5.3 // indirect
	github.com/google/gnostic-models v0.6.8 // indirect
	github.com/google/go-cmp v0.5.9 // indirect
	github.com/google/gofuzz v1.2.0 // indirect
	github.com/josharian/intern v1.0.0 // indirect
	github.com/json-iterator/go v1.1.12 // indirect
	github.com/mailru/easyjson v0.7.7 // indirect
	github.com/matttproud/golang_protobuf_extensions v1.0.4 // indirect
	github.com/modern-go/concurrent v0.0.0-20180306012644-bacd9c7ef1dd // indirect
	github.com/modern-go/reflect2 v1.0.2 // indirect
	github.com/pmezard/go-difflib v1.0.0 // indirect
	github.com/prometheus/client_model v0.4.0 // indirect
	github.com/prometheus/common v0.44.0
	github.com/prometheus/procfs v0.10.1 // indirect
	github.com/spf13/pflag v1.0.5 // indirect
	golang.org/x/net v0.17.0 // indirect
	golang.org/x/oauth2 v0.8.0 // indirect
	golang.org/x/sys v0.29.0 // indirect
	golang.org/x/term v0.13.0 // indirect
	golang.org/x/text v0.13.0 // indirect
	golang.org/x/time v0.3.0 // indirect
	google.golang.org/appengine v1.6.7 // indirect
	google.golang.org/protobuf v1.31.0 // indirect
	gopkg.in/inf.v0 v0.9.1 // indirect
	gopkg.in/yaml.v2 v2.4.0 // indirect
	gopkg.in/yaml.v3 v3.0.1 // indirect
	k8s.io/kube-openapi v0.0.0-20230717233707-2695361300d9 // indirect
	k8s.io/utils v0.0.0-20230726121419-3b25d923346b // indirect
	sigs.k8s.io/json v0.0.0-20221116044647-bc3834ca7abd // indirect
	sigs.k8s.io/structured-merge-diff/v4 v4.3.0 // indirect
	sigs.k8s.io/yaml v1.3.0 // indirect
)

require (
	gonum.org/v1/gonum v0.8.2
	k8s.io/api v0.24.16
	k8s.io/apimachinery v0.24.16
	k8s.io/client-go v0.24.16
)

require (
	github.com/dgryski/go-rendezvous v0.0.0-20200823014737-9f7001d12a5f // indirect
	github.com/evanphx/json-patch v5.6.0+incompatible // indirect
	github.com/fsnotify/fsnotify v1.5.4 // indirect
	github.com/golang/groupcache v0.0.0-20210331224755-41bb18bfe9da // indirect
	github.com/google/gnostic v0.5.7-v3refs // indirect
	github.com/google/uuid v1.3.0 // indirect
	github.com/munnerz/goautoneg v0.0.0-20191010083416-a7dc8b61c822 // indirect
	github.com/pkg/errors v0.9.1 // indirect
	github.com/stretchr/objx v0.5.0 // indirect
	go.uber.org/multierr v1.10.0 // indirect
	golang.org/x/exp v0.0.0-20220303212507-bbda1eaf7a17 // indirect
	k8s.io/apiserver v0.0.0 // indirect
	k8s.io/component-base v0.24.16 // indirect
	k8s.io/component-helpers v0.20.0-alpha.2 // indirect
	k8s.io/kube-scheduler v0.0.0 // indirect
)

replace github.com/google/gnostic/openapiv2 => github.com/google/gnostic/openapiv2 v0.0.0-20220911222732-39aa17dd7850

replace github.com/google/gnostic/openapiv3 => github.com/google/gnostic/openapiv3 v0.0.0-20220911222732-39aa17dd7850

replace k8s.io/component-base => k8s.io/component-base v0.24.16

replace k8s.io/apiextensions-apiserver => k8s.io/apiextensions-apiserver v0.24.16
