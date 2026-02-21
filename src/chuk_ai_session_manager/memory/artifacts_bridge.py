# chuk_ai_session_manager/memory/artifacts_bridge.py
"""
Artifacts Bridge for AI Virtual Memory.

The ArtifactsBridge integrates with chuk-artifacts for persistent page storage.
It handles storing/loading pages to/from the L3 and L4 tiers.

When chuk-artifacts is not installed, falls back to in-memory storage.

Design principles:
- Async-native: All I/O operations are async
- Optional dependency: Works without chuk-artifacts
- Pydantic-native: All models are BaseModel subclasses
- No magic strings: Uses constants for MIME types
"""

import logging
from datetime import datetime
from typing import Any, Protocol

from pydantic import BaseModel, Field

from chuk_ai_session_manager.exceptions import StorageError

from .models import (
    MEMORY_PAGE_MIME_TYPE,
    VM_CHECKPOINT_MIME_TYPE,
    MemoryPage,
    StorageStats,
    StorageTier,
)

logger = logging.getLogger(__name__)

# Try to import chuk-artifacts
try:
    from chuk_artifacts import ArtifactStore
    from chuk_artifacts.models import StorageScope

    ARTIFACTS_AVAILABLE = True
except ImportError:
    ARTIFACTS_AVAILABLE = False
    ArtifactStore = None  # type: ignore
    StorageScope = None  # type: ignore


class StorageBackend(Protocol):
    """Protocol for page storage backends."""

    async def store(
        self,
        page: MemoryPage,
        tier: StorageTier,
    ) -> str:
        """Store a page and return its artifact_id."""
        ...

    async def load(
        self,
        artifact_id: str,
    ) -> MemoryPage | None:
        """Load a page by artifact_id."""
        ...

    async def delete(
        self,
        artifact_id: str,
    ) -> bool:
        """Delete a page by artifact_id."""
        ...


class PageMetadata(BaseModel):
    """Metadata stored with a page in artifacts."""

    page_id: str
    modality: str
    compression_level: int
    tier: str
    session_id: str | None = None


class CheckpointMetadata(BaseModel):
    """Metadata stored with a checkpoint."""

    checkpoint_name: str
    page_count: int
    session_id: str | None = None


class CheckpointEntry(BaseModel):
    """Entry for a page in a checkpoint manifest."""

    page_id: str
    artifact_id: str
    modality: str


class CheckpointManifest(BaseModel):
    """Manifest for a checkpoint."""

    name: str
    created_at: datetime
    page_count: int
    pages: list[CheckpointEntry] = Field(default_factory=list)


class InMemoryBackend(BaseModel):
    """
    Simple in-memory storage backend for testing/development.

    Not persistent - pages are lost when process exits.
    """

    pages: dict[str, bytes] = Field(default_factory=dict)
    counter: int = Field(default=0)

    model_config = {"arbitrary_types_allowed": True}

    async def store(
        self,
        page: MemoryPage,
        tier: StorageTier,
    ) -> str:
        """Store a page in memory."""
        self.counter += 1
        artifact_id = f"mem_{tier.value}_{self.counter}"

        # Serialize page to JSON bytes
        data = page.model_dump_json().encode("utf-8")
        self.pages[artifact_id] = data

        return artifact_id

    async def load(
        self,
        artifact_id: str,
    ) -> MemoryPage | None:
        """Load a page from memory."""
        data = self.pages.get(artifact_id)
        if not data:
            return None

        # Deserialize
        return MemoryPage.model_validate_json(data)

    async def delete(
        self,
        artifact_id: str,
    ) -> bool:
        """Delete a page from memory."""
        if artifact_id in self.pages:
            del self.pages[artifact_id]
            return True
        return False

    def clear(self) -> None:
        """Clear all stored pages."""
        self.pages.clear()
        self.counter = 0


class ArtifactsBridge(BaseModel):
    """
    Bridge between VM and chuk-artifacts storage.

    Handles page serialization/deserialization and tier-appropriate storage.

    Usage:
        # With chuk-artifacts
        bridge = ArtifactsBridge()
        await bridge.configure(artifact_store)

        # Without chuk-artifacts (in-memory fallback)
        bridge = ArtifactsBridge()
        await bridge.configure()  # Uses InMemoryBackend

        # Store a page
        artifact_id = await bridge.store_page(page, StorageTier.L3)

        # Load a page
        page = await bridge.load_page(artifact_id)
    """

    # Configuration
    session_id: str | None = None
    _backend: InMemoryBackend | None = None
    _artifact_store: Any | None = None
    _using_artifacts: bool = False

    model_config = {"arbitrary_types_allowed": True}

    async def configure(
        self,
        artifact_store: Any | None = None,
        session_id: str | None = None,
    ) -> None:
        """
        Configure the bridge with a storage backend.

        Args:
            artifact_store: Optional ArtifactStore instance
            session_id: Optional session ID for scoping

        If no artifact_store is provided, uses in-memory backend.
        """
        self.session_id = session_id

        if artifact_store and not ARTIFACTS_AVAILABLE:
            raise ImportError(
                "chuk-artifacts is required for artifact bridge. Install with: pip install chuk-artifacts"
            )

        if artifact_store and ARTIFACTS_AVAILABLE:
            self._artifact_store = artifact_store
            self._using_artifacts = True
        else:
            self._backend = InMemoryBackend()
            self._using_artifacts = False

    @property
    def is_persistent(self) -> bool:
        """Check if using persistent storage."""
        return self._using_artifacts

    async def store_page(
        self,
        page: MemoryPage,
        tier: StorageTier = StorageTier.L3,
    ) -> str:
        """
        Store a page to the specified tier.

        Args:
            page: MemoryPage to store
            tier: Target storage tier (L3 or L4)

        Returns:
            Artifact ID for later retrieval
        """
        if self._using_artifacts and self._artifact_store:
            return await self._store_with_artifacts(page, tier)
        elif self._backend:
            return await self._backend.store(page, tier)
        else:
            raise StorageError("ArtifactsBridge not configured")

    async def _store_with_artifacts(
        self,
        page: MemoryPage,
        tier: StorageTier,
    ) -> str:
        """Store using chuk-artifacts."""
        if not ARTIFACTS_AVAILABLE:
            raise ImportError(
                "chuk-artifacts is required for artifact bridge. Install with: pip install chuk-artifacts"
            )
        # Serialize page
        data = page.model_dump_json().encode("utf-8")

        # Determine scope based on tier
        if StorageScope:
            scope = StorageScope.SESSION if tier == StorageTier.L3 else StorageScope.SANDBOX
        else:
            scope = "session"

        # Build metadata
        metadata = PageMetadata(
            page_id=page.page_id,
            modality=page.modality.value,
            compression_level=page.compression_level
            if isinstance(page.compression_level, int)
            else page.compression_level.value,
            tier=tier.value,
            session_id=self.session_id,
        )

        # Store in artifacts
        artifact_id = await self._artifact_store.store(
            data=data,
            mime=MEMORY_PAGE_MIME_TYPE,
            scope=scope,
            summary=f"VM page: {page.page_id}",
            metadata=metadata.model_dump(),
        )

        return artifact_id

    async def load_page(
        self,
        artifact_id: str,
    ) -> MemoryPage | None:
        """
        Load a page by artifact ID.

        Args:
            artifact_id: ID returned from store_page

        Returns:
            MemoryPage or None if not found
        """
        if self._using_artifacts and self._artifact_store:
            return await self._load_with_artifacts(artifact_id)
        elif self._backend:
            return await self._backend.load(artifact_id)
        else:
            raise StorageError("ArtifactsBridge not configured")

    async def _load_with_artifacts(
        self,
        artifact_id: str,
    ) -> MemoryPage | None:
        """Load using chuk-artifacts."""
        if not ARTIFACTS_AVAILABLE:
            raise ImportError(
                "chuk-artifacts is required for artifact bridge. Install with: pip install chuk-artifacts"
            )
        try:
            # Get artifact data
            data = await self._artifact_store.retrieve(artifact_id)
            if not data:
                return None

            # Deserialize
            if isinstance(data, bytes):
                return MemoryPage.model_validate_json(data)
            elif isinstance(data, str):
                return MemoryPage.model_validate_json(data.encode())
            else:
                return None
        except Exception:
            logger.warning("Failed to load artifact %s", artifact_id, exc_info=True)
            return None

    async def delete_page(
        self,
        artifact_id: str,
    ) -> bool:
        """
        Delete a stored page.

        Args:
            artifact_id: ID of the artifact to delete

        Returns:
            True if deleted, False if not found
        """
        if self._using_artifacts and self._artifact_store:
            try:
                await self._artifact_store.delete(artifact_id)
                return True
            except Exception:
                logger.warning("Failed to delete artifact %s", artifact_id, exc_info=True)
                return False
        elif self._backend:
            return await self._backend.delete(artifact_id)
        else:
            raise StorageError("ArtifactsBridge not configured")

    async def store_checkpoint(
        self,
        pages: list[MemoryPage],
        checkpoint_name: str,
    ) -> str:
        """
        Store a checkpoint of multiple pages.

        Args:
            pages: List of pages to checkpoint
            checkpoint_name: Name for the checkpoint

        Returns:
            Checkpoint artifact ID
        """
        # Create checkpoint manifest
        manifest = CheckpointManifest(
            name=checkpoint_name,
            created_at=datetime.utcnow(),
            page_count=len(pages),
            pages=[],
        )

        # Store each page and record artifact IDs
        for page in pages:
            artifact_id = await self.store_page(page, StorageTier.L3)
            manifest.pages.append(
                CheckpointEntry(
                    page_id=page.page_id,
                    artifact_id=artifact_id,
                    modality=page.modality.value,
                )
            )

        # Store manifest itself
        manifest_data = manifest.model_dump_json().encode("utf-8")

        if self._using_artifacts and self._artifact_store:
            scope = StorageScope.SESSION if StorageScope else "session"

            checkpoint_metadata = CheckpointMetadata(
                checkpoint_name=checkpoint_name,
                page_count=len(pages),
                session_id=self.session_id,
            )

            checkpoint_id = await self._artifact_store.store(
                data=manifest_data,
                mime=VM_CHECKPOINT_MIME_TYPE,
                scope=scope,
                summary=f"VM checkpoint: {checkpoint_name}",
                metadata=checkpoint_metadata.model_dump(),
            )
            return checkpoint_id
        else:
            # In-memory: store manifest directly
            if self._backend:
                self._backend.counter += 1
                checkpoint_id = f"checkpoint_{self._backend.counter}"
                self._backend.pages[checkpoint_id] = manifest_data
                return checkpoint_id
            raise StorageError("ArtifactsBridge not configured")

    async def load_checkpoint(
        self,
        checkpoint_id: str,
    ) -> list[MemoryPage]:
        """
        Load all pages from a checkpoint.

        Args:
            checkpoint_id: ID from store_checkpoint

        Returns:
            List of MemoryPages
        """
        # Load manifest
        manifest_data: bytes | None = None
        if self._using_artifacts and self._artifact_store:
            manifest_data = await self._artifact_store.retrieve(checkpoint_id)
        elif self._backend:
            manifest_data = self._backend.pages.get(checkpoint_id)
        else:
            raise StorageError("ArtifactsBridge not configured")

        if not manifest_data:
            return []

        manifest_str = manifest_data.decode("utf-8") if isinstance(manifest_data, bytes) else manifest_data

        manifest = CheckpointManifest.model_validate_json(manifest_str)

        # Load each page
        pages: list[MemoryPage] = []
        for entry in manifest.pages:
            page = await self.load_page(entry.artifact_id)
            if page:
                pages.append(page)

        return pages

    def get_stats(self) -> StorageStats:
        """Get storage statistics."""
        if self._using_artifacts:
            return StorageStats(
                backend="chuk-artifacts",
                persistent=True,
                session_id=self.session_id,
                pages_stored=0,  # Would need to query artifact store
            )
        elif self._backend:
            return StorageStats(
                backend="in-memory",
                persistent=False,
                session_id=self.session_id,
                pages_stored=len(self._backend.pages),
            )
        else:
            return StorageStats(
                backend="unconfigured",
                persistent=False,
            )
