# Practice 099 — Causal DAGs & Identification Strategies

D-separation and the backdoor/frontdoor criteria implemented from scratch, then used to explain why a confounder, a mediator, and a collider must be treated differently under conditioning — including simulated M-bias and selection bias where adding a control variable makes a correct model worse. Cross-checked against `DoWhy`'s identification API and `causal-learn`'s structure discovery. See [`CLAUDE.md`](./CLAUDE.md) for full instructions, theoretical context, and the command table.
