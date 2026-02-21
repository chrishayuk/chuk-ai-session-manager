# tests/test_demand_paging.py
"""
Tests for DemandPagingPrePass and MemoryManager demand-paging integration.

Covers:
- Recall signal detection (has_recall_signal)
- Topic extraction (extract_topics)
- Prefetch candidate generation (get_prefetch_candidates)
- MemoryManager integration (demand_pre_pass, search_pages, evict_segment_pages)
"""

from chuk_ai_session_manager.memory.demand_paging import DemandPagingPrePass
from chuk_ai_session_manager.memory.manager import MemoryManager
from chuk_ai_session_manager.memory.models import (
    PageType,
    StorageTier,
)
from chuk_ai_session_manager.memory.page_table import PageTable

# ===========================================================================
# TestRecallSignalDetection
# ===========================================================================


class TestRecallSignalDetection:
    """Test DemandPagingPrePass.has_recall_signal()."""

    def test_detects_what_did_we(self):
        pager = DemandPagingPrePass()
        assert pager.has_recall_signal("What did we discuss about auth?") is True

    def test_detects_remember_when(self):
        pager = DemandPagingPrePass()
        assert pager.has_recall_signal("Remember when we chose PostgreSQL?") is True

    def test_detects_remind_me(self):
        pager = DemandPagingPrePass()
        assert pager.has_recall_signal("Remind me about the API design") is True

    def test_detects_you_mentioned(self):
        pager = DemandPagingPrePass()
        assert pager.has_recall_signal("You mentioned something about caching") is True

    def test_no_signal_normal_message(self):
        pager = DemandPagingPrePass()
        assert pager.has_recall_signal("How do I sort a list?") is False

    def test_no_signal_empty(self):
        pager = DemandPagingPrePass()
        assert pager.has_recall_signal("") is False

    def test_custom_signals(self):
        custom = [r"(look up|find again)"]
        pager = DemandPagingPrePass(recall_signals=custom)
        # Custom signal should match
        assert pager.has_recall_signal("Can you look up that info?") is True
        # Default signals should NOT match (replaced by custom)
        assert pager.has_recall_signal("What did we discuss?") is False

    def test_detects_earlier(self):
        pager = DemandPagingPrePass()
        assert pager.has_recall_signal("As we discussed earlier") is True

    def test_detects_previously(self):
        pager = DemandPagingPrePass()
        assert pager.has_recall_signal("Previously we decided on Redis") is True

    def test_detects_go_back_to(self):
        pager = DemandPagingPrePass()
        assert pager.has_recall_signal("Let's go back to the architecture discussion") is True

    def test_detects_revisit(self):
        pager = DemandPagingPrePass()
        assert pager.has_recall_signal("Can we revisit the caching strategy?") is True

    def test_detects_you_said(self):
        pager = DemandPagingPrePass()
        assert pager.has_recall_signal("You said we should use FastAPI") is True

    def test_detects_you_suggested(self):
        pager = DemandPagingPrePass()
        assert pager.has_recall_signal("You suggested using Redis for caching") is True

    def test_detects_you_recommended(self):
        pager = DemandPagingPrePass()
        assert pager.has_recall_signal("You recommended a microservices approach") is True

    def test_detects_refresh_my_memory(self):
        pager = DemandPagingPrePass()
        assert pager.has_recall_signal("Refresh my memory on the auth flow") is True

    def test_detects_that_decision_about(self):
        pager = DemandPagingPrePass()
        assert pager.has_recall_signal("What about that decision about the database?") is True

    def test_detects_that_plan_for(self):
        pager = DemandPagingPrePass()
        assert pager.has_recall_signal("Can we reconsider that plan for deployment?") is True

    def test_detects_return_to(self):
        pager = DemandPagingPrePass()
        assert pager.has_recall_signal("I want to return to the earlier topic") is True

    def test_case_insensitive(self):
        pager = DemandPagingPrePass()
        assert pager.has_recall_signal("WHAT DID WE discuss about AUTH?") is True

    def test_no_signal_technical_question(self):
        pager = DemandPagingPrePass()
        assert pager.has_recall_signal("How do I configure nginx reverse proxy?") is False

    def test_no_signal_greeting(self):
        pager = DemandPagingPrePass()
        assert pager.has_recall_signal("Hello, how are you today?") is False


# ===========================================================================
# TestTopicExtraction
# ===========================================================================


class TestTopicExtraction:
    """Test DemandPagingPrePass.extract_topics()."""

    def test_extracts_meaningful_words(self):
        pager = DemandPagingPrePass()
        topics = pager.extract_topics("What about the database design?")
        assert "what" in topics
        assert "about" in topics
        assert "database" in topics
        assert "design" in topics

    def test_filters_short_words(self):
        pager = DemandPagingPrePass()
        topics = pager.extract_topics("I am the best")
        # "I" (1), "am" (2), "the" (3) all < 4 chars, only "best" (4) kept
        assert topics == ["best"]

    def test_empty_message(self):
        pager = DemandPagingPrePass()
        topics = pager.extract_topics("")
        assert topics == []

    def test_handles_punctuation(self):
        pager = DemandPagingPrePass()
        topics = pager.extract_topics("PostgreSQL, FastAPI, and Redis!")
        assert "postgresql" in topics
        assert "fastapi" in topics
        assert "redis" in topics
        # "and" (3 chars) should be filtered out
        assert "and" not in topics

    def test_lowercases_all_tokens(self):
        pager = DemandPagingPrePass()
        topics = pager.extract_topics("DATABASE DESIGN")
        assert "database" in topics
        assert "design" in topics
        # No uppercase tokens
        assert all(t == t.lower() for t in topics)

    def test_splits_on_special_characters(self):
        pager = DemandPagingPrePass()
        topics = pager.extract_topics("auth-flow and session_management")
        assert "auth" in topics
        assert "flow" in topics
        assert "session" in topics
        assert "management" in topics

    def test_keeps_alphanumeric(self):
        pager = DemandPagingPrePass()
        topics = pager.extract_topics("Using python3 with http2 support")
        assert "using" in topics
        assert "python3" in topics
        assert "http2" in topics
        assert "support" in topics

    def test_only_short_words_returns_empty(self):
        pager = DemandPagingPrePass()
        topics = pager.extract_topics("I am an ok")
        # All words < 4 chars
        assert topics == []

    def test_whitespace_only(self):
        pager = DemandPagingPrePass()
        topics = pager.extract_topics("   ")
        assert topics == []


# ===========================================================================
# TestPrefetchCandidates
# ===========================================================================


class TestPrefetchCandidates:
    """Test DemandPagingPrePass.get_prefetch_candidates()."""

    def _make_page_table_with_pages(self, pages_spec):
        """
        Helper to build a PageTable with registered pages.

        pages_spec: list of (page_id, page_type) tuples.
        Returns (page_table, page_hints dict).
        """
        from chuk_ai_session_manager.memory.models import MemoryPage, Modality

        pt = PageTable()
        hints = {}
        for page_id, page_type, hint in pages_spec:
            page = MemoryPage(
                page_id=page_id,
                session_id="test",
                content=f"Content for {page_id}",
                modality=Modality.TEXT,
                page_type=page_type,
                storage_tier=StorageTier.L2,
            )
            pt.register(page)
            if hint:
                hints[page_id] = hint
        return pt, hints

    def test_includes_claims_on_recall(self):
        """Recall signal + CLAIM pages not in L0 -> included."""
        pt, hints = self._make_page_table_with_pages(
            [
                ("claim_auth", PageType.CLAIM, "authentication decision"),
                ("claim_db", PageType.CLAIM, "database choice"),
                ("transcript_1", PageType.TRANSCRIPT, "some talk"),
            ]
        )
        pager = DemandPagingPrePass()
        candidates = pager.get_prefetch_candidates(
            message="What did we decide about auth?",
            page_table=pt,
            page_hints=hints,
            working_set_ids=set(),
        )
        assert "claim_auth" in candidates
        assert "claim_db" in candidates

    def test_includes_summaries_on_recall(self):
        """Recall signal + SUMMARY pages -> included."""
        pt, hints = self._make_page_table_with_pages(
            [
                ("summary_seg1", PageType.SUMMARY, "segment 1 summary"),
                ("transcript_1", PageType.TRANSCRIPT, "some talk"),
            ]
        )
        pager = DemandPagingPrePass()
        candidates = pager.get_prefetch_candidates(
            message="Remind me what we discussed",
            page_table=pt,
            page_hints=hints,
            working_set_ids=set(),
        )
        assert "summary_seg1" in candidates

    def test_excludes_pages_already_in_l0(self):
        """Pages already in working_set_ids should be excluded."""
        pt, hints = self._make_page_table_with_pages(
            [
                ("claim_auth", PageType.CLAIM, "auth decision"),
                ("claim_db", PageType.CLAIM, "db choice"),
            ]
        )
        pager = DemandPagingPrePass()
        candidates = pager.get_prefetch_candidates(
            message="What did we decide?",
            page_table=pt,
            page_hints=hints,
            working_set_ids={"claim_auth"},  # already in L0
        )
        assert "claim_auth" not in candidates
        assert "claim_db" in candidates

    def test_topic_matching(self):
        """Topics extracted from message match page hints."""
        pt, hints = self._make_page_table_with_pages(
            [
                ("page_db", PageType.TRANSCRIPT, "database design decisions"),
                ("page_api", PageType.TRANSCRIPT, "REST API endpoints"),
                ("page_auth", PageType.TRANSCRIPT, "authentication flow"),
            ]
        )
        pager = DemandPagingPrePass()
        candidates = pager.get_prefetch_candidates(
            message="Tell me about the database",
            page_table=pt,
            page_hints=hints,
            working_set_ids=set(),
        )
        # "database" matches hint "database design decisions"
        assert "page_db" in candidates
        # "page_api" hint does not match any topic in the message
        assert "page_api" not in candidates

    def test_no_recall_no_claims(self):
        """Without recall signal, claims are NOT automatically included."""
        pt, hints = self._make_page_table_with_pages(
            [
                ("claim_auth", PageType.CLAIM, "auth decision"),
                ("page_db", PageType.TRANSCRIPT, "database info"),
            ]
        )
        pager = DemandPagingPrePass()
        candidates = pager.get_prefetch_candidates(
            message="Tell me about database",  # no recall signal
            page_table=pt,
            page_hints=hints,
            working_set_ids=set(),
        )
        # claim_auth should NOT be included (no recall signal, no topic match)
        assert "claim_auth" not in candidates
        # page_db should be included via topic matching ("database" in hint)
        assert "page_db" in candidates

    def test_max_prefetch_limit(self):
        """More candidates than max_prefetch_pages -> capped."""
        pages_spec = [(f"claim_{i}", PageType.CLAIM, f"claim {i}") for i in range(10)]
        pt, hints = self._make_page_table_with_pages(pages_spec)

        pager = DemandPagingPrePass(max_prefetch_pages=3)
        candidates = pager.get_prefetch_candidates(
            message="What did we decide?",  # recall signal
            page_table=pt,
            page_hints=hints,
            working_set_ids=set(),
        )
        assert len(candidates) <= 3

    def test_empty_message(self):
        """Empty message -> no candidates."""
        pt, hints = self._make_page_table_with_pages(
            [
                ("claim_auth", PageType.CLAIM, "auth decision"),
            ]
        )
        pager = DemandPagingPrePass()
        candidates = pager.get_prefetch_candidates(
            message="",
            page_table=pt,
            page_hints=hints,
            working_set_ids=set(),
        )
        assert candidates == []

    def test_case_insensitive_hints(self):
        """Topic matching is case-insensitive against hints."""
        pt, hints = self._make_page_table_with_pages(
            [
                ("page_db", PageType.TRANSCRIPT, "Database Design Decisions"),
            ]
        )
        pager = DemandPagingPrePass()
        candidates = pager.get_prefetch_candidates(
            message="Tell me about the DATABASE",
            page_table=pt,
            page_hints=hints,
            working_set_ids=set(),
        )
        # "database" (lowered from "DATABASE") should match "Database Design Decisions" (lowered)
        assert "page_db" in candidates

    def test_deduplication(self):
        """Pages should not appear twice even if matched by both claim and topic."""
        pt, hints = self._make_page_table_with_pages(
            [
                ("claim_db", PageType.CLAIM, "database choice"),
            ]
        )
        pager = DemandPagingPrePass()
        candidates = pager.get_prefetch_candidates(
            message="What did we decide about the database?",  # recall + topic
            page_table=pt,
            page_hints=hints,
            working_set_ids=set(),
        )
        # claim_db matched as CLAIM (via recall) AND via topic ("database")
        # but should only appear once
        assert candidates.count("claim_db") == 1

    def test_no_pages_in_table(self):
        """Empty page table returns no candidates."""
        pt = PageTable()
        pager = DemandPagingPrePass()
        candidates = pager.get_prefetch_candidates(
            message="What did we discuss earlier?",
            page_table=pt,
            page_hints={},
            working_set_ids=set(),
        )
        assert candidates == []

    def test_all_pages_in_working_set(self):
        """When all pages are already in L0, no candidates returned."""
        pt, hints = self._make_page_table_with_pages(
            [
                ("claim_1", PageType.CLAIM, "claim one"),
                ("summary_1", PageType.SUMMARY, "summary one"),
            ]
        )
        pager = DemandPagingPrePass()
        candidates = pager.get_prefetch_candidates(
            message="What did we decide earlier?",
            page_table=pt,
            page_hints=hints,
            working_set_ids={"claim_1", "summary_1"},
        )
        assert candidates == []

    def test_max_prefetch_pages_default(self):
        """Default max_prefetch_pages is 5."""
        pager = DemandPagingPrePass()
        assert pager._max_prefetch_pages == 5

    def test_custom_max_prefetch_pages(self):
        """Custom max_prefetch_pages is respected."""
        pager = DemandPagingPrePass(max_prefetch_pages=10)
        assert pager._max_prefetch_pages == 10


# ===========================================================================
# TestDemandPagingIntegration
# ===========================================================================


class TestDemandPagingIntegration:
    """Integration tests using MemoryManager with demand paging."""

    async def test_demand_pre_pass_faults_pages(self):
        """Create pages, evict some, then demand_pre_pass faults them back."""
        vm = MemoryManager(session_id="dp-test")

        # Create claim and summary pages, add to working set
        claim = vm.create_page(
            "We decided to use PostgreSQL",
            page_type=PageType.CLAIM,
            page_id="claim_pg",
            size_tokens=20,
            hint="postgresql database decision",
        )
        await vm.add_to_working_set(claim)

        summary = vm.create_page(
            "Summary of segment 1: discussed databases",
            page_type=PageType.SUMMARY,
            page_id="summary_seg1",
            size_tokens=30,
            hint="segment 1 database discussion",
        )
        await vm.add_to_working_set(summary)

        # Verify both in L0
        l0_ids = vm.working_set.get_l0_page_ids()
        assert "claim_pg" in l0_ids
        assert "summary_seg1" in l0_ids

        # Evict them to L2
        await vm.evict_page("claim_pg", StorageTier.L2)
        await vm.evict_page("summary_seg1", StorageTier.L2)

        # Verify evicted from L0
        l0_ids = vm.working_set.get_l0_page_ids()
        assert "claim_pg" not in l0_ids
        assert "summary_seg1" not in l0_ids

        # Run demand pre-pass with a recall message
        faulted = await vm.demand_pre_pass("What did we decide about PostgreSQL?")

        # Pages should be faulted back into L0
        assert len(faulted) > 0
        l0_ids = vm.working_set.get_l0_page_ids()
        # At least one of the pages should be back in L0
        assert any(pid in l0_ids for pid in ["claim_pg", "summary_seg1"])

    async def test_demand_pre_pass_empty_message(self):
        """Empty message triggers no faults."""
        vm = MemoryManager(session_id="dp-empty")
        vm.create_page(
            "Some claim",
            page_type=PageType.CLAIM,
            page_id="claim_x",
            size_tokens=10,
            hint="claim about something",
        )

        faulted = await vm.demand_pre_pass("")
        assert faulted == []

    async def test_demand_pre_pass_no_recall_no_faults_for_claims(self):
        """Normal message without recall signal does not fault claim pages."""
        vm = MemoryManager(session_id="dp-normal")

        # Create a claim but do NOT add to working set (stays in L1/page_store)
        vm.create_page(
            "Auth decision",
            page_type=PageType.CLAIM,
            page_id="claim_auth",
            size_tokens=10,
            hint="authentication decision",
        )

        # Non-recall message, no topic match on "authentication"
        faulted = await vm.demand_pre_pass("How do I sort a list in Python?")
        assert "claim_auth" not in faulted

    async def test_demand_pre_pass_topic_match_faults(self):
        """Topic matching faults pages by hint, even without recall signal."""
        vm = MemoryManager(session_id="dp-topic")

        vm.create_page(
            "Database design notes",
            page_type=PageType.TRANSCRIPT,
            page_id="notes_db",
            size_tokens=15,
            hint="database design architecture",
        )
        # Page starts in L1 (not in working set L0)
        # It's in the page store and page table from create_page

        faulted = await vm.demand_pre_pass("Tell me about the database design")
        # "database" and "design" should match the hint
        assert "notes_db" in faulted

    async def test_search_pages(self):
        """Create pages with hints, search by query."""
        vm = MemoryManager(session_id="search-test")

        vm.create_page(
            "PostgreSQL schema design",
            page_id="pg_schema",
            size_tokens=10,
            hint="database schema postgresql",
        )
        vm.create_page(
            "Redis caching strategy",
            page_id="redis_cache",
            size_tokens=10,
            hint="caching redis performance",
        )
        vm.create_page(
            "Auth flow diagram",
            page_id="auth_flow",
            size_tokens=10,
            hint="authentication oauth flow",
        )

        result = await vm.search_pages("database")
        assert len(result.results) >= 1
        page_ids = [r.page_id for r in result.results]
        assert "pg_schema" in page_ids

    async def test_search_pages_no_match(self):
        """Search for non-existent topic returns empty results."""
        vm = MemoryManager(session_id="search-empty")
        vm.create_page(
            "Some content",
            page_id="p1",
            size_tokens=10,
            hint="weather forecast",
        )

        result = await vm.search_pages("quantum physics")
        assert len(result.results) == 0

    async def test_search_pages_respects_limit(self):
        """Search results respect the limit parameter."""
        vm = MemoryManager(session_id="search-limit")
        for i in range(10):
            vm.create_page(
                f"Database page {i}",
                page_id=f"db_{i}",
                size_tokens=10,
                hint=f"database topic {i}",
            )

        result = await vm.search_pages("database", limit=3)
        assert len(result.results) <= 3

    async def test_evict_segment_pages(self):
        """Evict segment pages: non-pinned evicted, pinned stays."""
        vm = MemoryManager(session_id="evict-seg")

        # Create and add pages to working set
        t1 = vm.create_page("Turn 1", page_id="t1", size_tokens=10)
        await vm.add_to_working_set(t1)

        t2 = vm.create_page("Turn 2", page_id="t2", size_tokens=10)
        await vm.add_to_working_set(t2)

        pinned = vm.create_page(
            "Important claim",
            page_type=PageType.CLAIM,
            page_id="pinned_claim",
            size_tokens=10,
        )
        await vm.add_to_working_set(pinned)
        vm.pin_page("pinned_claim")

        # Verify all three in L0
        l0_ids = vm.working_set.get_l0_page_ids()
        assert "t1" in l0_ids
        assert "t2" in l0_ids
        assert "pinned_claim" in l0_ids

        # Evict segment pages
        evicted = await vm.evict_segment_pages(StorageTier.L2)

        # Non-pinned pages should be evicted
        assert "t1" in evicted
        assert "t2" in evicted
        # Pinned page should NOT be evicted
        assert "pinned_claim" not in evicted

        # Verify working set state
        l0_ids = vm.working_set.get_l0_page_ids()
        assert "t1" not in l0_ids
        assert "t2" not in l0_ids
        assert "pinned_claim" in l0_ids

    async def test_evict_segment_pages_empty_working_set(self):
        """Evicting when working set is empty returns empty list."""
        vm = MemoryManager(session_id="evict-empty")
        evicted = await vm.evict_segment_pages()
        assert evicted == []

    async def test_evict_segment_pages_all_pinned(self):
        """When all pages are pinned, nothing is evicted."""
        vm = MemoryManager(session_id="evict-pinned")

        p1 = vm.create_page("Claim 1", page_id="c1", size_tokens=10)
        await vm.add_to_working_set(p1)
        vm.pin_page("c1")

        p2 = vm.create_page("Claim 2", page_id="c2", size_tokens=10)
        await vm.add_to_working_set(p2)
        vm.pin_page("c2")

        evicted = await vm.evict_segment_pages()
        assert evicted == []

        l0_ids = vm.working_set.get_l0_page_ids()
        assert "c1" in l0_ids
        assert "c2" in l0_ids

    async def test_evict_segment_updates_page_table(self):
        """Evicted pages should have their tier updated in the page table."""
        vm = MemoryManager(session_id="evict-pt")

        p = vm.create_page("Transcript", page_id="tx1", size_tokens=10)
        await vm.add_to_working_set(p)

        await vm.evict_segment_pages(StorageTier.L2)

        entry = vm.page_table.lookup("tx1")
        assert entry is not None
        assert entry.tier == StorageTier.L2

    async def test_demand_pre_pass_respects_fault_policy(self):
        """Demand pre-pass respects the fault policy limits."""
        from chuk_ai_session_manager.memory.models import FaultPolicy

        policy = FaultPolicy(max_faults_per_turn=1)
        vm = MemoryManager(session_id="dp-policy", fault_policy=policy)

        # Create multiple claim pages
        for i in range(5):
            vm.create_page(
                f"Claim {i}",
                page_type=PageType.CLAIM,
                page_id=f"claim_{i}",
                size_tokens=10,
                hint=f"decision {i}",
            )

        # Run demand pre-pass with recall signal
        faulted = await vm.demand_pre_pass("What did we decide earlier?")
        # Fault policy limits to 1 fault per turn (but policy also has token limit)
        # At minimum we know it won't fault all 5
        assert len(faulted) <= policy.max_faults_per_turn

    async def test_full_lifecycle_create_evict_recall(self):
        """Full lifecycle: create pages, add to L0, evict, recall via demand paging."""
        vm = MemoryManager(session_id="lifecycle")

        # Turn 1: User asks about databases, we record a claim
        vm.new_turn()
        user_msg = vm.create_page(
            "Should we use PostgreSQL?",
            page_type=PageType.TRANSCRIPT,
            page_id="user_t1",
            size_tokens=10,
        )
        await vm.add_to_working_set(user_msg)

        claim = vm.create_page(
            "Decision: Use PostgreSQL for the main database",
            page_type=PageType.CLAIM,
            page_id="claim_postgres",
            size_tokens=15,
            hint="postgresql database decision",
        )
        await vm.add_to_working_set(claim)

        # Turn 2: Segment rollover, evict old transcript but pin claim
        vm.new_turn()
        vm.pin_page("claim_postgres")
        evicted = await vm.evict_segment_pages(StorageTier.L2)
        assert "user_t1" in evicted
        assert "claim_postgres" not in evicted

        # Turn 3: User recalls the database decision
        vm.new_turn()
        await vm.demand_pre_pass("What did we decide about the database?")
        # claim_postgres was pinned and still in L0, so it won't be faulted
        # (it was never evicted)
        l0_ids = vm.working_set.get_l0_page_ids()
        assert "claim_postgres" in l0_ids
