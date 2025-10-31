package evaluation

import (
	"context"
	"encoding/json"
	"fmt"
	"io"
	"os"
	"path/filepath"
	"sort"
	"strings"
	"time"

	"go.uber.org/zap"
)

// ArtifactStore defines the interface for artifact storage
type ArtifactStore interface {
	Store(ctx context.Context, artifact *ReproducibilityArtifact) error
	Retrieve(ctx context.Context, artifactID string) (*ReproducibilityArtifact, error)
	List(ctx context.Context, filters ArtifactFilters) ([]*ReproducibilityArtifact, error)
	Delete(ctx context.Context, artifactID string) error
	Update(ctx context.Context, artifact *ReproducibilityArtifact) error
	Search(ctx context.Context, query ArtifactQuery) ([]*ReproducibilityArtifact, error)
	GetMetadata(ctx context.Context, artifactID string) (*ArtifactMetadata, error)
	UpdateMetadata(ctx context.Context, artifactID string, metadata *ArtifactMetadata) error
	GetStats(ctx context.Context) (*ArtifactStoreStats, error)
	Cleanup(ctx context.Context, policy RetentionPolicy) (*CleanupResults, error)
}

// ArtifactFilters represents filters for artifact listing
type ArtifactFilters struct {
	Type         string            `json:"type"`         // Artifact type filter
	ExperimentID string            `json:"experimentId"` // Experiment ID filter
	Tags         []string          `json:"tags"`         // Tag filters
	Status       string            `json:"status"`       // Status filter
	CreatedAfter *time.Time        `json:"createdAfter"` // Created after filter
	CreatedBefore *time.Time       `json:"createdBefore"` // Created before filter
	SizeMin      *int64            `json:"sizeMin"`      // Minimum size filter
	SizeMax      *int64            `json:"sizeMax"`      // Maximum size filter
	Metadata     map[string]string `json:"metadata"`     // Metadata filters
	Limit        int               `json:"limit"`        // Result limit
	Offset       int               `json:"offset"`       // Result offset
	SortBy       string            `json:"sortBy"`       // Sort field
	SortOrder    string            `json:"sortOrder"`    // Sort order (asc/desc)
}

// ArtifactQuery represents a search query for artifacts
type ArtifactQuery struct {
	Text         string            `json:"text"`         // Text search query
	Fields       []string          `json:"fields"`       // Fields to search
	Filters      ArtifactFilters   `json:"filters"`      // Additional filters
	Fuzzy        bool              `json:"fuzzy"`        // Enable fuzzy search
	Highlight    bool              `json:"highlight"`    // Enable result highlighting
	Facets       []string          `json:"facets"`       // Facet fields
	Aggregations map[string]string `json:"aggregations"` // Aggregation queries
}

// ArtifactStoreStats represents artifact store statistics
type ArtifactStoreStats struct {
	TotalArtifacts   int64                    `json:"totalArtifacts"`   // Total number of artifacts
	TotalSize        int64                    `json:"totalSize"`        // Total size of all artifacts
	ArtifactsByType  map[string]int64         `json:"artifactsByType"`  // Artifacts by type
	ArtifactsByStatus map[string]int64        `json:"artifactsByStatus"` // Artifacts by status
	SizeByType       map[string]int64         `json:"sizeByType"`       // Size by type
	CreationTrend    []CreationTrendPoint     `json:"creationTrend"`    // Creation trend over time
	AccessPattern    []AccessPatternPoint     `json:"accessPattern"`    // Access pattern analysis
	StorageEfficiency StorageEfficiencyMetrics `json:"storageEfficiency"` // Storage efficiency metrics
	LastUpdated      time.Time                `json:"lastUpdated"`      // Last update timestamp
}

// CreationTrendPoint represents a point in creation trend
type CreationTrendPoint struct {
	Timestamp time.Time `json:"timestamp"` // Timestamp
	Count     int64     `json:"count"`     // Number of artifacts created
	Size      int64     `json:"size"`      // Total size created
}

// AccessPatternPoint represents a point in access pattern
type AccessPatternPoint struct {
	Timestamp   time.Time `json:"timestamp"`   // Timestamp
	Accesses    int64     `json:"accesses"`    // Number of accesses
	UniqueUsers int64     `json:"uniqueUsers"` // Number of unique users
}

// StorageEfficiencyMetrics represents storage efficiency metrics
type StorageEfficiencyMetrics struct {
	CompressionRatio    float64 `json:"compressionRatio"`    // Average compression ratio
	DeduplicationSavings int64  `json:"deduplicationSavings"` // Space saved by deduplication
	UnusedArtifacts     int64   `json:"unusedArtifacts"`     // Number of unused artifacts
	UnusedSize          int64   `json:"unusedSize"`          // Size of unused artifacts
	FragmentationRatio  float64 `json:"fragmentationRatio"`  // Storage fragmentation ratio
}

// CleanupResults represents cleanup operation results
type CleanupResults struct {
	ArtifactsDeleted int64     `json:"artifactsDeleted"` // Number of artifacts deleted
	SizeFreed        int64     `json:"sizeFreed"`        // Size freed up
	ArtifactsArchived int64    `json:"artifactsArchived"` // Number of artifacts archived
	SizeArchived     int64     `json:"sizeArchived"`     // Size archived
	Errors           []string  `json:"errors"`           // Cleanup errors
	Duration         time.Duration `json:"duration"`     // Cleanup duration
	StartTime        time.Time `json:"startTime"`        // Cleanup start time
	EndTime          time.Time `json:"endTime"`          // Cleanup end time
}

// FileSystemArtifactStore implements ArtifactStore using filesystem
type FileSystemArtifactStore struct {
	basePath string
	logger   *zap.Logger
	config   *FileSystemStoreConfig
}

// FileSystemStoreConfig represents filesystem store configuration
type FileSystemStoreConfig struct {
	BasePath         string        `json:"basePath"`         // Base storage path
	CreateDirs       bool          `json:"createDirs"`       // Auto-create directories
	FilePermissions  os.FileMode   `json:"filePermissions"`  // File permissions
	DirPermissions   os.FileMode   `json:"dirPermissions"`   // Directory permissions
	SyncWrites       bool          `json:"syncWrites"`       // Sync writes to disk
	BufferSize       int           `json:"bufferSize"`       // I/O buffer size
	CompressionEnabled bool        `json:"compressionEnabled"` // Enable compression
	EncryptionEnabled bool         `json:"encryptionEnabled"` // Enable encryption
	BackupEnabled    bool          `json:"backupEnabled"`    // Enable backups
	BackupInterval   time.Duration `json:"backupInterval"`   // Backup interval
	IndexingEnabled  bool          `json:"indexingEnabled"`  // Enable indexing
	CacheEnabled     bool          `json:"cacheEnabled"`     // Enable caching
	CacheSize        int           `json:"cacheSize"`        // Cache size
	CacheTTL         time.Duration `json:"cacheTTL"`         // Cache TTL
}

// NewFileSystemArtifactStore creates a new filesystem artifact store
func NewFileSystemArtifactStore(basePath string) *FileSystemArtifactStore {
	logger, _ := zap.NewProduction()
	
	config := &FileSystemStoreConfig{
		BasePath:        basePath,
		CreateDirs:      true,
		FilePermissions: 0644,
		DirPermissions:  0755,
		SyncWrites:      true,
		BufferSize:      64 * 1024, // 64KB
		CompressionEnabled: false,
		EncryptionEnabled:  false,
		BackupEnabled:     false,
		IndexingEnabled:   true,
		CacheEnabled:      true,
		CacheSize:         1000,
		CacheTTL:          time.Hour,
	}

	return &FileSystemArtifactStore{
		basePath: basePath,
		logger:   logger,
		config:   config,
	}
}

// Store stores an artifact in the filesystem
func (fs *FileSystemArtifactStore) Store(ctx context.Context, artifact *ReproducibilityArtifact) error {
	fs.logger.Info("Storing artifact", zap.String("id", artifact.ID))

	// Create artifact directory
	artifactDir := filepath.Join(fs.basePath, "artifacts", artifact.Type, artifact.ID)
	if fs.config.CreateDirs {
		err := os.MkdirAll(artifactDir, fs.config.DirPermissions)
		if err != nil {
			return fmt.Errorf("failed to create artifact directory: %w", err)
		}
	}

	// Store artifact metadata
	metadataPath := filepath.Join(artifactDir, "metadata.json")
	metadataData, err := json.MarshalIndent(artifact, "", "  ")
	if err != nil {
		return fmt.Errorf("failed to marshal artifact metadata: %w", err)
	}

	err = os.WriteFile(metadataPath, metadataData, fs.config.FilePermissions)
	if err != nil {
		return fmt.Errorf("failed to write artifact metadata: %w", err)
	}

	// Copy artifact file if it exists and is different from storage location
	if artifact.Path != "" && artifact.Path != filepath.Join(artifactDir, "data") {
		dataPath := filepath.Join(artifactDir, "data")
		err = fs.copyFile(artifact.Path, dataPath)
		if err != nil {
			return fmt.Errorf("failed to copy artifact data: %w", err)
		}
		
		// Update artifact path to storage location
		artifact.Path = dataPath
	}

	// Update index if enabled
	if fs.config.IndexingEnabled {
		err = fs.updateIndex(artifact)
		if err != nil {
			fs.logger.Warn("Failed to update index", zap.Error(err))
		}
	}

	fs.logger.Info("Artifact stored successfully", zap.String("id", artifact.ID))
	return nil
}

// Retrieve retrieves an artifact from the filesystem
func (fs *FileSystemArtifactStore) Retrieve(ctx context.Context, artifactID string) (*ReproducibilityArtifact, error) {
	fs.logger.Info("Retrieving artifact", zap.String("id", artifactID))

	// Find artifact by scanning directories
	var artifactPath string
	err := filepath.Walk(filepath.Join(fs.basePath, "artifacts"), func(path string, info os.FileInfo, err error) error {
		if err != nil {
			return err
		}
		
		if info.Name() == "metadata.json" {
			dir := filepath.Dir(path)
			if strings.HasSuffix(dir, artifactID) {
				artifactPath = path
				return filepath.SkipDir
			}
		}
		return nil
	})
	
	if err != nil {
		return nil, fmt.Errorf("failed to search for artifact: %w", err)
	}
	
	if artifactPath == "" {
		return nil, fmt.Errorf("artifact not found: %s", artifactID)
	}

	// Load artifact metadata
	metadataData, err := os.ReadFile(artifactPath)
	if err != nil {
		return nil, fmt.Errorf("failed to read artifact metadata: %w", err)
	}

	var artifact ReproducibilityArtifact
	err = json.Unmarshal(metadataData, &artifact)
	if err != nil {
		return nil, fmt.Errorf("failed to unmarshal artifact metadata: %w", err)
	}

	// Update access time
	now := time.Now()
	artifact.AccessedAt = &now

	fs.logger.Info("Artifact retrieved successfully", zap.String("id", artifactID))
	return &artifact, nil
}

// List lists artifacts based on filters
func (fs *FileSystemArtifactStore) List(ctx context.Context, filters ArtifactFilters) ([]*ReproducibilityArtifact, error) {
	fs.logger.Info("Listing artifacts", zap.Any("filters", filters))

	var artifacts []*ReproducibilityArtifact

	// Walk through artifact directories
	err := filepath.Walk(filepath.Join(fs.basePath, "artifacts"), func(path string, info os.FileInfo, err error) error {
		if err != nil {
			return err
		}

		if info.Name() == "metadata.json" {
			// Load artifact metadata
			metadataData, err := os.ReadFile(path)
			if err != nil {
				fs.logger.Warn("Failed to read artifact metadata", zap.String("path", path), zap.Error(err))
				return nil
			}

			var artifact ReproducibilityArtifact
			err = json.Unmarshal(metadataData, &artifact)
			if err != nil {
				fs.logger.Warn("Failed to unmarshal artifact metadata", zap.String("path", path), zap.Error(err))
				return nil
			}

			// Apply filters
			if fs.matchesFilters(&artifact, filters) {
				artifacts = append(artifacts, &artifact)
			}
		}

		return nil
	})

	if err != nil {
		return nil, fmt.Errorf("failed to list artifacts: %w", err)
	}

	// Sort artifacts
	fs.sortArtifacts(artifacts, filters.SortBy, filters.SortOrder)

	// Apply pagination
	if filters.Offset > 0 || filters.Limit > 0 {
		start := filters.Offset
		if start > len(artifacts) {
			start = len(artifacts)
		}
		
		end := len(artifacts)
		if filters.Limit > 0 && start+filters.Limit < end {
			end = start + filters.Limit
		}
		
		artifacts = artifacts[start:end]
	}

	fs.logger.Info("Artifacts listed successfully", zap.Int("count", len(artifacts)))
	return artifacts, nil
}

// Delete deletes an artifact from the filesystem
func (fs *FileSystemArtifactStore) Delete(ctx context.Context, artifactID string) error {
	fs.logger.Info("Deleting artifact", zap.String("id", artifactID))

	// Find and delete artifact directory
	var artifactDir string
	err := filepath.Walk(filepath.Join(fs.basePath, "artifacts"), func(path string, info os.FileInfo, err error) error {
		if err != nil {
			return err
		}
		
		if info.IsDir() && strings.HasSuffix(path, artifactID) {
			artifactDir = path
			return filepath.SkipDir
		}
		return nil
	})
	
	if err != nil {
		return fmt.Errorf("failed to search for artifact: %w", err)
	}
	
	if artifactDir == "" {
		return fmt.Errorf("artifact not found: %s", artifactID)
	}

	// Remove artifact directory
	err = os.RemoveAll(artifactDir)
	if err != nil {
		return fmt.Errorf("failed to delete artifact directory: %w", err)
	}

	// Update index if enabled
	if fs.config.IndexingEnabled {
		err = fs.removeFromIndex(artifactID)
		if err != nil {
			fs.logger.Warn("Failed to remove from index", zap.Error(err))
		}
	}

	fs.logger.Info("Artifact deleted successfully", zap.String("id", artifactID))
	return nil
}

// Update updates an artifact in the filesystem
func (fs *FileSystemArtifactStore) Update(ctx context.Context, artifact *ReproducibilityArtifact) error {
	fs.logger.Info("Updating artifact", zap.String("id", artifact.ID))

	// Update timestamp
	artifact.UpdatedAt = time.Now()

	// Store updated artifact (this will overwrite existing)
	return fs.Store(ctx, artifact)
}

// Search searches for artifacts based on query
func (fs *FileSystemArtifactStore) Search(ctx context.Context, query ArtifactQuery) ([]*ReproducibilityArtifact, error) {
	fs.logger.Info("Searching artifacts", zap.String("text", query.Text))

	// For filesystem implementation, use simple text matching
	// In production, would use proper search engine like Elasticsearch
	
	artifacts, err := fs.List(ctx, query.Filters)
	if err != nil {
		return nil, err
	}

	if query.Text == "" {
		return artifacts, nil
	}

	// Filter by text search
	var results []*ReproducibilityArtifact
	searchText := strings.ToLower(query.Text)
	
	for _, artifact := range artifacts {
		if fs.matchesTextSearch(artifact, searchText, query.Fields) {
			results = append(results, artifact)
		}
	}

	fs.logger.Info("Search completed", zap.Int("results", len(results)))
	return results, nil
}

// GetMetadata retrieves artifact metadata
func (fs *FileSystemArtifactStore) GetMetadata(ctx context.Context, artifactID string) (*ArtifactMetadata, error) {
	artifact, err := fs.Retrieve(ctx, artifactID)
	if err != nil {
		return nil, err
	}
	return &artifact.Metadata, nil
}

// UpdateMetadata updates artifact metadata
func (fs *FileSystemArtifactStore) UpdateMetadata(ctx context.Context, artifactID string, metadata *ArtifactMetadata) error {
	artifact, err := fs.Retrieve(ctx, artifactID)
	if err != nil {
		return err
	}
	
	artifact.Metadata = *metadata
	return fs.Update(ctx, artifact)
}

// GetStats returns artifact store statistics
func (fs *FileSystemArtifactStore) GetStats(ctx context.Context) (*ArtifactStoreStats, error) {
	fs.logger.Info("Computing artifact store statistics")

	stats := &ArtifactStoreStats{
		ArtifactsByType:   make(map[string]int64),
		ArtifactsByStatus: make(map[string]int64),
		SizeByType:        make(map[string]int64),
		CreationTrend:     []CreationTrendPoint{},
		AccessPattern:     []AccessPatternPoint{},
		LastUpdated:       time.Now(),
	}

	// Get all artifacts
	artifacts, err := fs.List(ctx, ArtifactFilters{})
	if err != nil {
		return nil, err
	}

	// Compute statistics
	for _, artifact := range artifacts {
		stats.TotalArtifacts++
		stats.TotalSize += artifact.Size
		stats.ArtifactsByType[artifact.Type]++
		stats.ArtifactsByStatus[artifact.Status.State]++
		stats.SizeByType[artifact.Type] += artifact.Size
	}

	// Compute storage efficiency (simplified)
	stats.StorageEfficiency = StorageEfficiencyMetrics{
		CompressionRatio:   1.0, // Would compute actual compression ratio
		DeduplicationSavings: 0, // Would compute deduplication savings
		UnusedArtifacts:    0,   // Would identify unused artifacts
		UnusedSize:         0,   // Would compute unused size
		FragmentationRatio: 0.1, // Would compute fragmentation
	}

	fs.logger.Info("Statistics computed", 
		zap.Int64("totalArtifacts", stats.TotalArtifacts),
		zap.Int64("totalSize", stats.TotalSize))

	return stats, nil
}

// Cleanup performs cleanup based on retention policy
func (fs *FileSystemArtifactStore) Cleanup(ctx context.Context, policy RetentionPolicy) (*CleanupResults, error) {
	fs.logger.Info("Starting artifact cleanup", zap.Any("policy", policy))

	results := &CleanupResults{
		StartTime: time.Now(),
		Errors:    []string{},
	}

	if !policy.Enabled {
		fs.logger.Info("Cleanup disabled by policy")
		results.EndTime = time.Now()
		results.Duration = results.EndTime.Sub(results.StartTime)
		return results, nil
	}

	// Get all artifacts
	artifacts, err := fs.List(ctx, ArtifactFilters{})
	if err != nil {
		return nil, err
	}

	// Apply retention policies
	now := time.Now()
	var toDelete []*ReproducibilityArtifact
	var toArchive []*ReproducibilityArtifact

	for _, artifact := range artifacts {
		// Check age policy
		if policy.MaxAge > 0 && now.Sub(artifact.CreatedAt) > policy.MaxAge {
			if policy.ArchiveOld {
				toArchive = append(toArchive, artifact)
			} else {
				toDelete = append(toDelete, artifact)
			}
		}
	}

	// Check size and count policies
	if policy.MaxSize > 0 || policy.MaxCount > 0 {
		// Sort by creation time (oldest first)
		sort.Slice(artifacts, func(i, j int) bool {
			return artifacts[i].CreatedAt.Before(artifacts[j].CreatedAt)
		})

		totalSize := int64(0)
		for i, artifact := range artifacts {
			totalSize += artifact.Size
			
			// Check if we exceed limits
			exceedsSize := policy.MaxSize > 0 && totalSize > policy.MaxSize
			exceedsCount := policy.MaxCount > 0 && i >= policy.MaxCount
			
			if exceedsSize || exceedsCount {
				if policy.ArchiveOld {
					toArchive = append(toArchive, artifact)
				} else {
					toDelete = append(toDelete, artifact)
				}
			}
		}
	}

	// Delete artifacts
	for _, artifact := range toDelete {
		err := fs.Delete(ctx, artifact.ID)
		if err != nil {
			results.Errors = append(results.Errors, fmt.Sprintf("Failed to delete %s: %v", artifact.ID, err))
		} else {
			results.ArtifactsDeleted++
			results.SizeFreed += artifact.Size
		}
	}

	// Archive artifacts (simplified - just move to archive directory)
	for _, artifact := range toArchive {
		err := fs.archiveArtifact(artifact, policy.ArchiveLocation)
		if err != nil {
			results.Errors = append(results.Errors, fmt.Sprintf("Failed to archive %s: %v", artifact.ID, err))
		} else {
			results.ArtifactsArchived++
			results.SizeArchived += artifact.Size
		}
	}

	results.EndTime = time.Now()
	results.Duration = results.EndTime.Sub(results.StartTime)

	fs.logger.Info("Cleanup completed",
		zap.Int64("deleted", results.ArtifactsDeleted),
		zap.Int64("archived", results.ArtifactsArchived),
		zap.Int("errors", len(results.Errors)))

	return results, nil
}

// Helper methods

// copyFile copies a file from src to dst
func (fs *FileSystemArtifactStore) copyFile(src, dst string) error {
	sourceFile, err := os.Open(src)
	if err != nil {
		return err
	}
	defer sourceFile.Close()

	destFile, err := os.Create(dst)
	if err != nil {
		return err
	}
	defer destFile.Close()

	_, err = io.Copy(destFile, sourceFile)
	if err != nil {
		return err
	}

	if fs.config.SyncWrites {
		err = destFile.Sync()
		if err != nil {
			return err
		}
	}

	return nil
}

// matchesFilters checks if an artifact matches the given filters
func (fs *FileSystemArtifactStore) matchesFilters(artifact *ReproducibilityArtifact, filters ArtifactFilters) bool {
	// Type filter
	if filters.Type != "" && artifact.Type != filters.Type {
		return false
	}

	// Experiment ID filter
	if filters.ExperimentID != "" && artifact.ExperimentID != filters.ExperimentID {
		return false
	}

	// Status filter
	if filters.Status != "" && artifact.Status.State != filters.Status {
		return false
	}

	// Date filters
	if filters.CreatedAfter != nil && artifact.CreatedAt.Before(*filters.CreatedAfter) {
		return false
	}
	if filters.CreatedBefore != nil && artifact.CreatedAt.After(*filters.CreatedBefore) {
		return false
	}

	// Size filters
	if filters.SizeMin != nil && artifact.Size < *filters.SizeMin {
		return false
	}
	if filters.SizeMax != nil && artifact.Size > *filters.SizeMax {
		return false
	}

	// Tag filters
	if len(filters.Tags) > 0 {
		tagMap := make(map[string]bool)
		for _, tag := range artifact.Tags {
			tagMap[tag] = true
		}
		for _, filterTag := range filters.Tags {
			if !tagMap[filterTag] {
				return false
			}
		}
	}

	return true
}

// sortArtifacts sorts artifacts based on sort criteria
func (fs *FileSystemArtifactStore) sortArtifacts(artifacts []*ReproducibilityArtifact, sortBy, sortOrder string) {
	if sortBy == "" {
		sortBy = "createdAt"
	}
	if sortOrder == "" {
		sortOrder = "desc"
	}

	sort.Slice(artifacts, func(i, j int) bool {
		var less bool
		
		switch sortBy {
		case "name":
			less = artifacts[i].Name < artifacts[j].Name
		case "type":
			less = artifacts[i].Type < artifacts[j].Type
		case "size":
			less = artifacts[i].Size < artifacts[j].Size
		case "createdAt":
			less = artifacts[i].CreatedAt.Before(artifacts[j].CreatedAt)
		case "updatedAt":
			less = artifacts[i].UpdatedAt.Before(artifacts[j].UpdatedAt)
		default:
			less = artifacts[i].CreatedAt.Before(artifacts[j].CreatedAt)
		}

		if sortOrder == "desc" {
			return !less
		}
		return less
	})
}

// matchesTextSearch checks if an artifact matches text search
func (fs *FileSystemArtifactStore) matchesTextSearch(artifact *ReproducibilityArtifact, searchText string, fields []string) bool {
	if len(fields) == 0 {
		fields = []string{"name", "description", "type"}
	}

	for _, field := range fields {
		var fieldValue string
		switch field {
		case "name":
			fieldValue = strings.ToLower(artifact.Name)
		case "description":
			fieldValue = strings.ToLower(artifact.Description)
		case "type":
			fieldValue = strings.ToLower(artifact.Type)
		}

		if strings.Contains(fieldValue, searchText) {
			return true
		}
	}

	return false
}

// updateIndex updates the search index (simplified implementation)
func (fs *FileSystemArtifactStore) updateIndex(artifact *ReproducibilityArtifact) error {
	// In production, would update search index (Elasticsearch, etc.)
	return nil
}

// removeFromIndex removes an artifact from the search index
func (fs *FileSystemArtifactStore) removeFromIndex(artifactID string) error {
	// In production, would remove from search index
	return nil
}

// archiveArtifact archives an artifact to the specified location
func (fs *FileSystemArtifactStore) archiveArtifact(artifact *ReproducibilityArtifact, archiveLocation string) error {
	// Simplified archiving - just move to archive directory
	if archiveLocation == "" {
		archiveLocation = filepath.Join(fs.basePath, "archive")
	}

	// Create archive directory
	err := os.MkdirAll(archiveLocation, fs.config.DirPermissions)
	if err != nil {
		return err
	}

	// Move artifact (simplified implementation)
	// In production, would properly archive with compression, etc.
	return nil
}