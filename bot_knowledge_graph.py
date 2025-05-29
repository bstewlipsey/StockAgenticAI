import networkx as nx
import datetime

class KnowledgeGraphBot:
    """
    Manages a knowledge graph of trading decisions and outcomes for AI learning.
    Uses NetworkX for in-memory graph representation.
    """
    def __init__(self):
        self.graph = nx.DiGraph()

    def add_trading_decision(self, decision_id, symbol, action, confidence, timestamp=None, **kwargs):
        if not timestamp:
            timestamp = datetime.datetime.utcnow().isoformat()
        self.graph.add_node(
            f"decision_{decision_id}",
            type="TradingDecision",
            symbol=symbol,
            action=action,
            confidence=confidence,
            timestamp=timestamp,
            **kwargs
        )

    def add_trade_outcome(self, outcome_id, decision_id, result, pnl, timestamp=None, **kwargs):
        if not timestamp:
            timestamp = datetime.datetime.utcnow().isoformat()
        self.graph.add_node(
            f"outcome_{outcome_id}",
            type="TradeOutcome",
            result=result,
            pnl=pnl,
            timestamp=timestamp,
            **kwargs
        )
        # Link the decision to the outcome
        self.graph.add_edge(f"decision_{decision_id}", f"outcome_{outcome_id}", relation="RESULTED_IN")

    def get_decision_outcomes(self, decision_id):
        """Return all outcomes for a given decision_id."""
        node = f"decision_{decision_id}"
        return [self.graph.nodes[n] for n in self.graph.successors(node) if self.graph.nodes[n]["type"] == "TradeOutcome"]

    def get_all_decisions(self):
        return [n for n, d in self.graph.nodes(data=True) if d.get("type") == "TradingDecision"]

    def get_all_outcomes(self):
        return [n for n, d in self.graph.nodes(data=True) if d.get("type") == "TradeOutcome"]

    def selftest(self):
        print("\n--- Running KnowledgeGraphBot Self-Test ---")
        try:
            self.add_trading_decision(decision_id=1, symbol="AAPL", action="BUY", confidence=0.92)
            self.add_trade_outcome(outcome_id=101, decision_id=1, result="WIN", pnl=50.0)
            self.add_trading_decision(decision_id=2, symbol="TSLA", action="SELL", confidence=0.85)
            self.add_trade_outcome(outcome_id=102, decision_id=2, result="LOSS", pnl=-20.0)
            # Test retrieval
            outcomes = self.get_decision_outcomes(1)
            assert len(outcomes) == 1 and outcomes[0]["result"] == "WIN", "Outcome retrieval failed."
            assert len(self.get_all_decisions()) == 2, "Decision count incorrect."
            assert len(self.get_all_outcomes()) == 2, "Outcome count incorrect."
            print("--- KnowledgeGraphBot Self-Test PASSED ---")
        except Exception as e:
            print(f"--- KnowledgeGraphBot Self-Test FAILED: {e} ---")

if __name__ == "__main__":
    bot = KnowledgeGraphBot()
    bot.selftest()
