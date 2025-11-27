# GitHub Copilot Instructions

- Always generate TypeScript that compiles with `strict` mode enabled.
- Prefer `async/await` syntax over raw Promise chains for asynchronous flows.
- Use `zod` schemas to validate runtime inputs and outputs.
- Encourage functional programming patterns (pure functions, immutability, composition).
- Add JSDoc comments for every exported or public API surface.
- Include Vitest test coverage for all new features and edge cases.
- Choose descriptive, intention-revealing variable and function names.
