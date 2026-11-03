import Foundation
import HuggingFace

public enum ModelUtils {
    public static func resolveModelType(
        repoID: Repo.ID,
        hfToken: String? = nil,
        cache: HubCache = .default
    ) async throws -> String? {
        let modelNameComponents = repoID.name.split(separator: "/").last?.split(separator: "-")
        let configURL = try await resolveModelConfig(repoID: repoID, hfToken: hfToken, cache: cache)
        let configJSON = try JSONSerialization.jsonObject(with: Data(contentsOf: configURL))
        if let config = configJSON as? [String: Any] {
            return (config["model_type"] as? String)
                ?? (config["architecture"] as? String)
                ?? (config["model_version"] as? String)
                ?? (config["flow_lm"] != nil && config["mimi"] != nil ? "pocket_tts" : nil)
                ?? modelNameComponents?.first?.lowercased()
        }
        return nil
    }

    /// Resolves only config.json, including for models whose weights use a custom container.
    public static func resolveModelConfig(
        repoID: Repo.ID,
        hfToken: String? = nil,
        cache: HubCache = .default
    ) async throws -> URL {
        let modelSubdir = repoID.description.replacingOccurrences(of: "/", with: "_")
        let cachedConfig = cache.cacheDirectory.appendingPathComponent("mlx-audio/\(modelSubdir)/config.json")
        let configURL: URL
        if FileManager.default.fileExists(atPath: cachedConfig.path) {
            configURL = cachedConfig
        } else {
            let client: HubClient
            if let hfToken, !hfToken.isEmpty {
                client = HubClient(host: HubClient.defaultHost, bearerToken: hfToken, cache: cache)
            } else {
                client = HubClient(cache: cache)
            }
            configURL = try await client.downloadFile(at: "config.json", from: repoID)
        }
        return configURL
    }

    /// Resolves a model from cache or downloads it if not cached.
    /// - Parameters:
    ///   - string: The repository name
    ///   - requiredExtension: File extension that must exist for cache to be considered complete (e.g., "safetensors")
    ///   - hfToken: The huggingface token for access to gated repositories, if needed.
    /// - Returns: The model directory URL
    public static func resolveOrDownloadModel(
        repoID: Repo.ID,
        requiredExtension: String,
        additionalMatchingPatterns: [String] = [],
        hfToken: String? = nil,
        cache: HubCache = .default
    ) async throws -> URL {
        let client: HubClient
        if let token = hfToken, !token.isEmpty {
            print("Using HuggingFace token from configuration")
            client = HubClient(host: HubClient.defaultHost, bearerToken: token, cache: cache)
        } else {
            client = HubClient(cache: cache)
        }
        let resolvedCache = client.cache ?? cache
        return try await resolveOrDownloadModel(
            client: client,
            cache: resolvedCache,
            repoID: repoID,
            requiredExtension: requiredExtension,
            additionalMatchingPatterns: additionalMatchingPatterns
        )
    }

    /// Resolves a model from cache or downloads it if not cached.
    /// - Parameters:
    ///   - client: The HuggingFace Hub client
    ///   - cache: The HuggingFace cache
    ///   - repoID: The repository ID
    ///   - requiredExtension: File extension that must exist for cache to be considered complete (e.g., "safetensors")
    /// - Returns: The model directory URL
    public static func resolveOrDownloadModel(
        client: HubClient,
        cache: HubCache = .default,
        repoID: Repo.ID,
        requiredExtension: String,
        additionalMatchingPatterns: [String] = [],
        progressHandler: (@MainActor @Sendable (Progress) -> Void)? = nil
    ) async throws -> URL {
        let normalizedRequiredExtension = requiredExtension.hasPrefix(".")
            ? String(requiredExtension.dropFirst())
            : requiredExtension

        // Store downloaded model snapshots under the configured Hugging Face cache root.
        let modelSubdir = repoID.description.replacingOccurrences(of: "/", with: "_")
        let modelDir = cache.cacheDirectory
            .appendingPathComponent("mlx-audio")
            .appendingPathComponent(modelSubdir)

        let defaultPatterns: Set<String> = [
            "*.\(normalizedRequiredExtension)", "*.safetensors", "*.json", "*.txt", "*.wav",
        ]
        let matchingPatterns = defaultPatterns.union(additionalMatchingPatterns)
        let manifestURL = modelDir.appendingPathComponent(".mlx-audio-snapshot.json")
        var manifest = (try? Data(contentsOf: manifestURL)).flatMap {
            try? JSONDecoder().decode(SnapshotManifest.self, from: $0)
        }
        var hasUsableModel = false

        // Check if model already exists with required files
        if FileManager.default.fileExists(atPath: modelDir.path) {
            let files = try? FileManager.default.contentsOfDirectory(at: modelDir, includingPropertiesForKeys: [.fileSizeKey])
            let hasRequiredFile = files?.contains { file in
                guard file.pathExtension == normalizedRequiredExtension else { return false }
                let size = (try? file.resourceValues(forKeys: [.fileSizeKey]))?.fileSize ?? 0
                return size > 0
            } ?? false

            if hasRequiredFile {
                // Validate that config.json is valid JSON
                let configPath = modelDir.appendingPathComponent("config.json")
                if FileManager.default.fileExists(atPath: configPath.path) {
                    if let configData = try? Data(contentsOf: configPath),
                       let _ = try? JSONSerialization.jsonObject(with: configData) {
                        hasUsableModel = true
                        // Legacy caches already include the default patterns. Extra
                        // patterns need a repository listing, even if they match no files.
                        let isComplete = try manifest.map {
                            try missingFiles(in: $0, matching: matchingPatterns, modelDir: modelDir).isEmpty
                        } ?? Set(additionalMatchingPatterns).isSubset(of: defaultPatterns)
                        if isComplete {
                            print("Using cached model at: \(modelDir.path)")
                            return modelDir
                        }
                    } else {
                        print("Cached config.json is invalid, clearing cache...")
                        Self.clearCaches(modelDir: modelDir, repoID: repoID, hubCache: cache)
                    }
                } else if let manifest {
                    // Some repos ship no config.json (#242). The manifest lists every file
                    // of the snapshot, so it alone decides whether the cache is complete.
                    hasUsableModel = true
                    if try missingFiles(in: manifest, matching: matchingPatterns, modelDir: modelDir).isEmpty {
                        print("Using cached model at: \(modelDir.path)")
                        return modelDir
                    }
                }
            } else {
                print("Cached model appears incomplete, clearing cache...")
                Self.clearCaches(modelDir: modelDir, repoID: repoID, hubCache: cache)
            }
        }

        if !hasUsableModel {
            manifest = nil
            try? FileManager.default.removeItem(at: manifestURL)
        }

        // Create directory if needed
        try FileManager.default.createDirectory(at: modelDir, withIntermediateDirectories: true)

        if manifest == nil {
            // Pin the listing and downloads to the same commit, including when
            // completing a snapshot fetched by resolveModelType.
            let revision = hasUsableModel
                ? cache.resolveRevision(repo: repoID, kind: .model, ref: "main") ?? "main"
                : "main"
            let model = try await client.getModel(repoID, revision: revision, filesMetadata: true)
            guard let commit = model.sha, let files = model.siblings else {
                throw ModelUtilsError.missingFileListing(repoID.description)
            }
            manifest = SnapshotManifest(revision: commit, files: files)
        }
        guard let manifest else { throw ModelUtilsError.missingFileListing(repoID.description) }
        // Validate listed paths before passing them to the downloader.
        _ = try missingFiles(in: manifest, matching: matchingPatterns, modelDir: modelDir)

        print("Downloading model \(repoID)...")
        _ = try await client.downloadSnapshot(
            of: repoID,
            kind: .model,
            to: modelDir,
            revision: manifest.revision,
            matching: Array(matchingPatterns),
            progressHandler: progressHandler ?? { progress in
                print("\(progress.completedUnitCount)/\(progress.totalUnitCount) files")
            }
        )

        // Post-download validation: ensure required files are non-zero
        let downloadedFiles = try? FileManager.default.contentsOfDirectory(
            at: modelDir, includingPropertiesForKeys: [.fileSizeKey]
        )
        let hasValidFile = downloadedFiles?.contains { file in
            guard file.pathExtension == normalizedRequiredExtension else { return false }
            let size = (try? file.resourceValues(forKeys: [.fileSizeKey]))?.fileSize ?? 0
            return size > 0
        } ?? false

        if !hasValidFile {
            Self.clearCaches(modelDir: modelDir, repoID: repoID, hubCache: cache)
            throw ModelUtilsError.incompleteDownload(repoID.description)
        }

        // downloadSnapshot can return a partial Hub cache when offline. Only
        // the listing and the actual files together prove this request complete.
        let missing = try missingFiles(in: manifest, matching: matchingPatterns, modelDir: modelDir)
        guard missing.isEmpty else {
            throw ModelUtilsError.missingFiles(repoID.description, missing)
        }
        try JSONEncoder().encode(manifest).write(to: manifestURL, options: .atomic)

        print("Model downloaded to: \(modelDir.path)")
        return modelDir
    }

    private struct SnapshotManifest: Codable {
        let revision: String
        let files: [Model.SiblingInfo]
    }

    private static func missingFiles(
        in manifest: SnapshotManifest, matching patterns: Set<String>, modelDir: URL
    ) throws -> [String] {
        try manifest.files.compactMap { file in
            let path = file.relativeFilename
            guard patterns.contains(where: { fnmatch($0, path, 0) == 0 }) else { return nil }
            guard !path.isEmpty, !path.hasPrefix("/"), !path.contains("\\"), !path.contains("\0"),
                  !path.split(separator: "/").contains("..") else {
                throw ModelUtilsError.invalidFilePath(path)
            }
            let url = modelDir.appendingPathComponent(path)
            let values = try? url.resourceValues(forKeys: [.isRegularFileKey, .fileSizeKey])
            guard values?.isRegularFile == true, let size = values?.fileSize,
                  file.size.map({ size == $0 }) ?? (size > 0) else { return path }
            return nil
        }
    }

    /// Deletes a cached model snapshot and its HuggingFace Hub cache entries from disk.
    /// - Parameters:
    ///   - repoID: The repository ID of the model to delete.
    ///   - cache: The HuggingFace HubCache instance (defaults to `.default`).
    /// - Returns: `true` if any cached files or directories existed and were removed.
    @discardableResult
    public static func deleteCachedModel(
        repoID: Repo.ID,
        cache: HubCache = .default
    ) -> Bool {
        let rawDesc = repoID.description
        guard !rawDesc.isEmpty,
              !rawDesc.contains(".."),
              !rawDesc.contains("\\"),
              !rawDesc.contains("\0"),
              !rawDesc.hasPrefix("/") else {
            return false
        }
        let fm = FileManager.default
        let cacheRootPath = cache.cacheDirectory.standardizedFileURL.path
        let modelSubdir = rawDesc.replacingOccurrences(of: "/", with: "_")
        guard !modelSubdir.isEmpty, modelSubdir != ".", modelSubdir != ".." else {
            return false
        }
        let modelDir = cache.cacheDirectory
            .appendingPathComponent("mlx-audio")
            .appendingPathComponent(modelSubdir)
        let hubRepoDir = cache.repoDirectory(repo: repoID, kind: .model)
        let hubRepoName = hubRepoDir.lastPathComponent
        guard !hubRepoName.isEmpty, hubRepoName != ".", hubRepoName != "..", !hubRepoName.contains("..") else {
            return false
        }
        let metadataDir = cache.cacheDirectory.appendingPathComponent(".metadata").appendingPathComponent(hubRepoName)
        let locksDir = cache.cacheDirectory.appendingPathComponent(".locks").appendingPathComponent(hubRepoName)

        var removedAny = false

        // Remove any unpacked Phonon 2.5 container directory referenced by config.json
        for digest in phonon2ContainerDigests(modelDir: modelDir, hubRepoDir: hubRepoDir) {
            let unpackedDir = cache.cacheDirectory
                .appendingPathComponent("mlx-audio/phonon2")
                .appendingPathComponent(digest)
            let stdPath = unpackedDir.standardizedFileURL.path
            if stdPath.hasPrefix(cacheRootPath + "/"), fm.fileExists(atPath: stdPath) {
                if (try? fm.removeItem(at: unpackedDir)) != nil {
                    removedAny = true
                }
            }
        }

        for dir in [modelDir, hubRepoDir, metadataDir, locksDir] {
            let stdPath = dir.standardizedFileURL.path
            if stdPath.hasPrefix(cacheRootPath + "/"), fm.fileExists(atPath: stdPath) {
                if (try? fm.removeItem(at: dir)) != nil {
                    removedAny = true
                }
            }
        }
        return removedAny
    }

    private static func phonon2ContainerDigests(modelDir: URL, hubRepoDir: URL) -> Set<String> {
        var digests = Set<String>()
        var candidateConfigs: [URL] = [modelDir.appendingPathComponent("config.json")]
        let snapshotsDir = hubRepoDir.appendingPathComponent("snapshots")
        if let snaps = try? FileManager.default.contentsOfDirectory(at: snapshotsDir, includingPropertiesForKeys: nil) {
            for snap in snaps {
                candidateConfigs.append(snap.appendingPathComponent("config.json"))
            }
        }
        for configURL in candidateConfigs {
            guard let data = try? Data(contentsOf: configURL),
                  let json = try? JSONSerialization.jsonObject(with: data) as? [String: Any] else { continue }
            if let artifact = json["artifact"] as? [String: Any],
               let sha = artifact["sha256"] as? String, !sha.isEmpty,
               !sha.contains("/"), !sha.contains("..") {
                digests.insert(sha)
            }
            if let fermion = json["fermion"] as? [String: Any],
               let sha = fermion["container_sha256"] as? String, !sha.isEmpty,
               !sha.contains("/"), !sha.contains("..") {
                digests.insert(sha)
            }
        }
        return digests
    }

    private static func clearCaches(modelDir: URL, repoID: Repo.ID, hubCache: HubCache) {
        _ = deleteCachedModel(repoID: repoID, cache: hubCache)
    }
}

public enum ModelUtilsError: LocalizedError {
    case incompleteDownload(String)
    case missingFileListing(String)
    case missingFiles(String, [String])
    case invalidFilePath(String)

    public var errorDescription: String? {
        switch self {
        case .incompleteDownload(let repo):
            return "Downloaded model '\(repo)' has missing or zero-byte weight files. "
                + "The cache has been cleared — please try again."
        case .missingFileListing(let repo):
            return "The repository '\(repo)' did not provide a complete file listing."
        case .missingFiles(let repo, let files):
            return "Model '\(repo)' is missing or has incomplete files: \(files.joined(separator: ", ")). Connect to the Hub and try again."
        case .invalidFilePath(let path):
            return "The repository contains an invalid file path: '\(path)'."
        }
    }
}
