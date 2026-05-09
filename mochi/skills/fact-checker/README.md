# Fact Checker

Verify factual claims in documents using web search and official sources, then apply corrections with user confirmation.

## Features

- ✅ Comprehensive fact verification across multiple domains
- 🔍 Searches authoritative sources (official docs, API specs, academic papers)
- 📊 Generates detailed correction reports with sources
- 🤖 Auto-applies corrections after user approval
- 🕐 Adds temporal context to prevent information decay

## Supported Claim Types

- **AI Model Specifications**: Context windows, pricing, features, benchmarks
- **Technical Documentation**: API capabilities, version numbers, library features
- **Statistical Data**: Metrics, benchmark scores, performance data
- **General Facts**: Any verifiable factual statement

## Usage Examples

### Example 1: Update Outdated AI Model Info

```
User: Fact-check the AI model specifications in section 2.1
```

**What happens:**
1. Identifies claims: "Claude 3.5 Sonnet: 200K tokens", "GPT-4o: 128K tokens"
2. Searches official documentation for current models
3. Finds: Claude Sonnet 4.5, GPT-5.2 with updated specs
4. Generates correction report with sources
5. Applies fixes after user confirms

### Example 2: Verify Technical Claims

```
User: Check if these library versions are still current
```

**What happens:**
1. Extracts version numbers from document
2. Checks package registries (npm, PyPI, etc.)
3. Identifies outdated versions
4. Suggests updates with changelog references

### Example 3: Validate Statistics

```
User: Verify the benchmark scores in this section
```

**What happens:**
1. Identifies numerical claims and metrics
2. Searches official benchmark publications
3. Compares document values vs. source data
4. Flags discrepancies with authoritative links

## Workflow

The skill follows a 5-step process:

```
Fact-checking Progress:
- [ ] Step 1: Identify factual claims
- [ ] Step 2: Search authoritative sources
- [ ] Step 3: Compare claims against sources
- [ ] Step 4: Generate correction report
- [ ] Step 5: Apply corrections with user approval
```

## Source Evaluation

**Preferred sources (in order):**
1. Official product pages and documentation
2. API documentation and developer guides
3. Official blog announcements
4. GitHub releases (for open source)

**Use with caution:**
- Third-party aggregators (verify against official sources)
- Blog posts and articles (cross-reference)

**Avoid:**
- Outdated documentation
- Unofficial wikis without citations
- Speculation and rumors

## Real-World Example

**Before:**
```markdown
AI 大模型的"上下文窗口"不断升级：
- Claude 3.5 Sonnet: 200K tokens（约 15 万汉字）
- GPT-4o: 128K tokens（约 10 万汉字）
- Gemini 1.5 Pro: 2M tokens（约 150 万汉字）
```

**After fact-checking:**
```markdown
AI 大模型的"上下文窗口"不断升级（截至 2026 年 1 月）：
- Claude Sonnet 4.5: 200K tokens（约 15 万汉字）
- GPT-5.2: 400K tokens（约 30 万汉字）
- Gemini 3 Pro: 1M tokens（约 75 万汉字）
```

**Changes made:**
- ✅ Updated Claude 3.5 Sonnet → Claude Sonnet 4.5
- ✅ Corrected GPT-4o (128K) → GPT-5.2 (400K)
- ✅ Fixed Gemini 1.5 Pro (2M) → Gemini 3 Pro (1M)
- ✅ Added temporal marker "截至 2026 年 1 月"

## Installation

```bash
# Via CCPM (recommended)
ccpm install @daymade-skills/fact-checker

# Manual installation
Download fact-checker.zip and install through Claude Code
```

## Trigger Keywords

The skill activates when you mention:
- "fact-check this document"
- "verify these claims"
- "check if this is accurate"
- "update outdated information"
- "validate the data"

## Configuration

No configuration required. The skill works out of the box.

## Limitations

**Cannot verify:**
- Subjective opinions or judgments
- Future predictions or specifications
- Claims requiring paywalled sources
- Disputed facts without authoritative consensus

**For such cases**, the skill will:
- Note the limitation in the report
- Suggest qualification language
- Recommend user research or expert consultation

## Best Practices

### For Authors

1. **Run regularly**: Fact-check documents periodically to catch outdated info
2. **Include dates**: Add temporal markers like "as of [date]" to claims
3. **Cite sources**: Keep original source links for future verification
4. **Review reports**: Always review the correction report before applying changes

### For Fact-Checking

1. **Be specific**: Target specific sections rather than entire books
2. **Verify critical claims first**: Prioritize high-impact information
3. **Cross-reference**: For important claims, verify across multiple sources
4. **Update regularly**: Technical specs change frequently - recheck periodically

## Development

Created with skill-creator v1.2.2 following Anthropic's best practices.

**Testing:**
- Verified on Claude Sonnet 4.5, Opus 4.5, and Haiku 4
- Tested with real-world documentation updates
- Validated correction workflow with user approval gates

## Version History

### 1.0.0 (2026-01-05)
- Initial release
- Support for AI models, technical docs, statistics
- Auto-correction with user approval
- Comprehensive source evaluation framework

## License

MIT License - See repository for details

## Contributing

Issues and pull requests welcome at [daymade/claude-code-skills](https://github.com/daymade/claude-code-skills)
