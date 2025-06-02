import networkx as nx
import datetime
from utils.logger_mixin import LoggerMixin


class KnowledgeGraphBot(LoggerMixin):
    """
    Manages a knowledge graph of trading decisions and outcomes for AI learning.
    Uses NetworkX for in-memory graph representation.
    Provides methods to add decisions, outcomes, and query relationships.
    """

    def __init__(self):
        """
        Initialize the knowledge graph as a directed graph.
        """
        super().__init__()
        self.graph = nx.DiGraph()

    def add_trading_decision(
        self, decision_id, symbol, action, confidence, timestamp=None, **kwargs
    ):
        """
        Add a trading decision node to the graph with relevant attributes.
        Links to outcomes can be added later.

        :param decision_id: Unique identifier for the trading decision.
        :param symbol: The stock or asset symbol.
        :param action: Action to be taken, e.g., 'BUY' or 'SELL'.
        :param confidence: Confidence level in the decision (0 to 1 scale).
        :param timestamp: Optional timestamp for when the decision was made.
        """
        method = "add_trading_decision"
        self.logger.debug(
            f"{self.__module__.split('.')[-1]} [{self.__class__.__name__}]: [{method}(decision_id={decision_id}, symbol={symbol}, action={action}, confidence={confidence})] START"
        )
        if not timestamp:
            timestamp = datetime.datetime.utcnow().isoformat()
        self.graph.add_node(
            f"decision_{decision_id}",
            type="TradingDecision",
            symbol=symbol,
            action=action,
            confidence=confidence,
            timestamp=timestamp,
            **kwargs,
        )
        self.logger.debug(
            f"{self.__module__.split('.')[-1]} [{self.__class__.__name__}]: [{method}(decision_id={decision_id}, symbol={symbol}, action={action}, confidence={confidence})] END"
        )

    def add_trade_outcome(
        self, outcome_id, decision_id, result, pnl, timestamp=None, **kwargs
    ):
        """
        Add a trade outcome node and link it to the corresponding decision node.
        Stores result, pnl, and timestamp.

        :param outcome_id: Unique identifier for the trade outcome.
        :param decision_id: The decision ID this outcome is linked to.
        :param result: The result of the trade, e.g., 'WIN' or 'LOSS'.
        :param pnl: Profit and loss amount from the trade.
        :param timestamp: Optional timestamp for when the outcome was recorded.
        """
        method = "add_trade_outcome"
        self.logger.debug(
            f"{self.__module__.split('.')[-1]} [{self.__class__.__name__}]: [{method}(outcome_id={outcome_id}, decision_id={decision_id}, result={result}, pnl={pnl})] START"
        )
        if not timestamp:
            timestamp = datetime.datetime.utcnow().isoformat()
        self.graph.add_node(
            f"outcome_{outcome_id}",
            type="TradeOutcome",
            result=result,
            pnl=pnl,
            timestamp=timestamp,
            **kwargs,
        )
        # Link the decision to the outcome
        self.graph.add_edge(
            f"decision_{decision_id}", f"outcome_{outcome_id}", relation="RESULTED_IN"
        )
        self.logger.debug(
            f"{self.__module__.split('.')[-1]} [{self.__class__.__name__}]: [{method}(outcome_id={outcome_id}, decision_id={decision_id}, result={result}, pnl={pnl})] END"
        )

    def get_decision_outcomes(self, decision_id):
        """
        Get all trade outcome nodes linked to a given decision node.
        Returns a list of outcome node attributes.

        :param decision_id: The decision ID to query outcomes for.
        :return: List of outcome attributes linked to the decision.
        """
        method = "get_decision_outcomes"
        self.logger.debug(
            f"{self.__module__.split('.')[-1]} [{self.__class__.__name__}]: [{method}(decision_id={decision_id})] START"
        )
        node = f"decision_{decision_id}"
        outcomes = [
            self.graph.nodes[n]
            for n in self.graph.successors(node)
            if self.graph.nodes[n]["type"] == "TradeOutcome"
        ]
        self.logger.debug(
            f"{self.__module__.split('.')[-1]} [{self.__class__.__name__}]: [{method}(decision_id={decision_id})] END"
        )
        return outcomes

    def get_all_decisions(self):
        """
        Get all decision nodes in the graph.
        Returns a list of node identifiers.

        :return: List of all decision node IDs.
        """
        method = "get_all_decisions"
        self.logger.debug(
            f"{self.__module__.split('.')[-1]} [{self.__class__.__name__}]: [{method}()] START"
        )
        decisions = [
            n
            for n, d in self.graph.nodes(data=True)
            if d.get("type") == "TradingDecision"
        ]
        self.logger.debug(
            f"{self.__module__.split('.')[-1]} [{self.__class__.__name__}]: [{method}()] END"
        )
        return decisions

    def get_all_outcomes(self):
        """
        Get all outcome nodes in the graph.
        Returns a list of node identifiers.

        :return: List of all outcome node IDs.
        """
        method = "get_all_outcomes"
        self.logger.debug(
            f"{self.__module__.split('.')[-1]} [{self.__class__.__name__}]: [{method}()] START"
        )
        outcomes = [
            n for n, d in self.graph.nodes(data=True) if d.get("type") == "TradeOutcome"
        ]
        self.logger.debug(
            f"{self.__module__.split('.')[-1]} [{self.__class__.__name__}]: [{method}()] END"
        )
        return outcomes

    def selftest(self):
        """
        Perform a self-test of the KnowledgeGraphBot functionality.
        Adds sample data and checks if retrieval methods work as expected.
        """
        print("\n--- Running KnowledgeGraphBot Self-Test ---")
        try:
            self.add_trading_decision(
                decision_id=1, symbol="AAPL", action="BUY", confidence=0.92
            )
            self.add_trade_outcome(
                outcome_id=101, decision_id=1, result="WIN", pnl=50.0
            )
            self.add_trading_decision(
                decision_id=2, symbol="TSLA", action="SELL", confidence=0.85
            )
            self.add_trade_outcome(
                outcome_id=102, decision_id=2, result="LOSS", pnl=-20.0
            )
            # Test retrieval
            outcomes = self.get_decision_outcomes(1)
            assert (
                len(outcomes) == 1 and outcomes[0]["result"] == "WIN"
            ), "Outcome retrieval failed."
            assert len(self.get_all_decisions()) == 2, "Decision count incorrect."
            assert len(self.get_all_outcomes()) == 2, "Outcome count incorrect."
            print("--- KnowledgeGraphBot Self-Test PASSED ---")
        except Exception as e:
            print(f"--- KnowledgeGraphBot Self-Test FAILED: {e} ---")


if __name__ == "__main__":
    bot = KnowledgeGraphBot()
    bot.selftest()
