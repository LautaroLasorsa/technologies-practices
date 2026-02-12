#include "quickfix/Application.h"
#include "quickfix/MessageCracker.h"
#include "quickfix/Session.h"
#include "quickfix/FileStore.h"
#include "quickfix/FileLog.h"
#include "quickfix/SocketInitiator.h"
#include "quickfix/SessionSettings.h"
#include "quickfix/SessionID.h"

// FIX 4.4 message types
#include "quickfix/fix44/NewOrderSingle.h"
#include "quickfix/fix44/ExecutionReport.h"
#include "quickfix/fix44/MessageCracker.h"

#include <iostream>
#include <string>
#include <thread>
#include <chrono>

// =============================================================================
// Minimal client application -- receives ExecutionReport responses
// =============================================================================

/// Simple FIX initiator (trading client) that connects to the mock exchange,
/// sends a NewOrderSingle, and prints the ExecutionReport response.
class ClientApplication
    : public FIX::Application
    , public FIX44::MessageCracker
{
public:
    void onCreate(const FIX::SessionID& id) override {
        std::cout << "[Client] Session created: " << id << "\n";
        session_id_ = id;
    }

    void onLogon(const FIX::SessionID& id) override {
        std::cout << "[Client] Logged on: " << id << "\n";
        logged_on_ = true;
    }

    void onLogout(const FIX::SessionID& id) override {
        std::cout << "[Client] Logged out: " << id << "\n";
        logged_on_ = false;
    }

    void toAdmin(FIX::Message&, const FIX::SessionID&) override {}
    void fromAdmin(const FIX::Message&, const FIX::SessionID&) override {}
    void toApp(FIX::Message&, const FIX::SessionID&) override {}

    void fromApp(const FIX::Message& message,
                 const FIX::SessionID& session_id) override {
        crack(message, session_id);
    }

    /// Called when the exchange sends back an ExecutionReport.
    void onMessage(const FIX44::ExecutionReport& report,
                   const FIX::SessionID& /*session_id*/) override {
        FIX::OrderID order_id;
        FIX::ExecType exec_type;
        FIX::OrdStatus ord_status;
        FIX::Symbol symbol;
        FIX::Side side;
        FIX::LeavesQty leaves_qty;
        FIX::CumQty cum_qty;

        report.get(order_id);
        report.get(exec_type);
        report.get(ord_status);
        report.get(symbol);
        report.get(side);
        report.get(leaves_qty);
        report.get(cum_qty);

        std::cout << "\n[Client] === Execution Report Received ===\n"
                  << "  OrderID:   " << order_id << "\n"
                  << "  ExecType:  " << exec_type << "\n"
                  << "  OrdStatus: " << ord_status << "\n"
                  << "  Symbol:    " << symbol << "\n"
                  << "  Side:      " << side << "\n"
                  << "  LeavesQty: " << leaves_qty << "\n"
                  << "  CumQty:    " << cum_qty << "\n"
                  << "==========================================\n";
    }

    bool is_logged_on() const { return logged_on_; }
    const FIX::SessionID& session_id() const { return session_id_; }

private:
    FIX::SessionID session_id_;
    bool logged_on_ = false;
};

// =============================================================================
// Build and send a NewOrderSingle
// =============================================================================

/// Build a FIX44::NewOrderSingle message for a limit order.
///
/// A NewOrderSingle (MsgType=D) is the standard way to submit an order
/// in the FIX protocol. It requires several mandatory fields.
FIX44::NewOrderSingle build_new_order(
    const std::string& symbol,
    char side,         // '1' = Buy, '2' = Sell
    double quantity,
    double price)
{
    // ── Exercise Context ──────────────────────────────────────────────────
    // This exercise teaches FIX order message construction from the client side.
    // NewOrderSingle is the most common FIX message type in electronic trading.
    // Understanding required vs optional fields (HandlInst, TimeInForce) and their
    // semantics is critical for order routing correctness in production systems.
    // ──────────────────────────────────────────────────────────────────────

    // TODO(human): Build a FIX44::NewOrderSingle message.
    //
    // The constructor requires:
    //   FIX::ClOrdID     -- client-assigned order ID (use any unique string, e.g., "order_001")
    //   FIX::Side        -- FIX::Side(side)  where side is '1' for Buy, '2' for Sell
    //   FIX::TransactTime -- FIX::TransactTime()  (defaults to now)
    //   FIX::OrdType     -- FIX::OrdType(FIX::OrdType_LIMIT) for a limit order
    //
    // After construction, set these additional fields:
    //   order.set(FIX::HandlInst('1'));           // Automated, no intervention
    //   order.set(FIX::Symbol(symbol));            // e.g., "AAPL"
    //   order.set(FIX::OrderQty(quantity));         // e.g., 100
    //   order.set(FIX::Price(price));               // e.g., 150.25
    //   order.set(FIX::TimeInForce(FIX::TimeInForce_DAY));  // Day order
    //
    // Return the constructed message.
    //
    // Reference: https://github.com/quickfix/quickfix/blob/master/examples/tradeclient/Application.cpp

    // --- Your code here ---
    // Replace the line below with your implementation:
    return FIX44::NewOrderSingle(
        FIX::ClOrdID("PLACEHOLDER"),
        FIX::Side(side),
        FIX::TransactTime(),
        FIX::OrdType(FIX::OrdType_LIMIT)
    );
}

/// Send the order to the exchange via the active FIX session.
void send_order(const FIX44::NewOrderSingle& order,
                const FIX::SessionID& session_id)
{
    // ── Exercise Context ──────────────────────────────────────────────────
    // This exercise teaches FIX message transmission via QuickFIX sessions.
    // sendToTarget abstracts sequencing, checksums, and header generation.
    // Understanding that QuickFIX queues messages asynchronously (not blocking)
    // is key for avoiding race conditions in production trading systems.
    // ──────────────────────────────────────────────────────────────────────

    // TODO(human): Send the message using FIX::Session::sendToTarget.
    //
    // Hint: FIX::Session::sendToTarget(order, session_id);
    //
    // This is a static method that looks up the session by ID and queues
    // the message for sending. QuickFIX handles sequencing, checksums,
    // and the FIX header automatically.

    // --- Your code here ---
}

// =============================================================================
// Main -- connect and send an order
// =============================================================================

int main(int argc, char* argv[])
{
    try {
        std::string config_path = "config/initiator.cfg";
        if (argc > 1) {
            config_path = argv[1];
        }

        FIX::SessionSettings settings(config_path);
        FIX::FileStoreFactory store_factory(settings);
        FIX::FileLogFactory log_factory(settings);

        ClientApplication application;

        // SocketInitiator actively connects to the acceptor.
        FIX::SocketInitiator initiator(application, store_factory, settings, log_factory);

        std::cout << "[Client] Connecting to exchange...\n";
        initiator.start();

        // Wait for logon to complete (up to 5 seconds).
        for (int i = 0; i < 50 && !application.is_logged_on(); ++i) {
            std::this_thread::sleep_for(std::chrono::milliseconds(100));
        }

        if (!application.is_logged_on()) {
            std::cerr << "[Client] Failed to log on. Is the acceptor running?\n";
            initiator.stop();
            return 1;
        }

        // Send a test order: Buy 100 shares of AAPL at $150.25
        std::cout << "\n[Client] Sending NewOrderSingle: Buy 100 AAPL @ 150.25\n";
        auto order = build_new_order("AAPL", FIX::Side_BUY, 100.0, 150.25);
        send_order(order, application.session_id());

        // Wait for the ExecutionReport response.
        std::this_thread::sleep_for(std::chrono::seconds(2));

        std::cout << "\n[Client] Press Enter to disconnect.\n";
        std::cin.get();

        initiator.stop();
        std::cout << "[Client] Disconnected.\n";

    } catch (const FIX::ConfigError& e) {
        std::cerr << "[Client] Configuration error: " << e.what() << "\n";
        return 1;
    } catch (const std::exception& e) {
        std::cerr << "[Client] Error: " << e.what() << "\n";
        return 1;
    }

    return 0;
}
