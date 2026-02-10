#pragma once

// QuickFIX headers
#include "quickfix/Application.h"
#include "quickfix/MessageCracker.h"
#include "quickfix/Session.h"
#include "quickfix/SessionID.h"
#include "quickfix/fix44/NewOrderSingle.h"
#include "quickfix/fix44/ExecutionReport.h"

// FIX field types used in our messages
#include "quickfix/fix44/MessageCracker.h"

#include <string>
#include <iostream>
#include <atomic>

/// Mock exchange application that receives NewOrderSingle messages
/// and responds with ExecutionReport acknowledgments.
///
/// Inherits from:
///   - FIX::Application      -> session lifecycle callbacks
///   - FIX44::MessageCracker -> type-safe message dispatch (cracks generic
///                              Message into specific FIX44 message types)
///
/// Pattern: QuickFIX routes incoming app messages through fromApp() -> crack()
///          -> onMessage(specific_type). This is the Visitor pattern applied
///          to FIX message types.
class FixApplication
    : public FIX::Application
    , public FIX44::MessageCracker
{
public:
    FixApplication() = default;

    // -------------------------------------------------------------------------
    // FIX::Application interface -- session lifecycle
    // -------------------------------------------------------------------------

    /// Called when a new session is created (before logon).
    void onCreate(const FIX::SessionID& session_id) override;

    /// Called when a counterparty successfully logs on.
    void onLogon(const FIX::SessionID& session_id) override;

    /// Called when a counterparty logs out or the session disconnects.
    void onLogout(const FIX::SessionID& session_id) override;

    /// Called before an admin message is sent (e.g., Logon, Heartbeat).
    /// You can modify the message here (e.g., add password to Logon).
    void toAdmin(FIX::Message& message,
                 const FIX::SessionID& session_id) override;

    /// Called when an admin message is received.
    void fromAdmin(const FIX::Message& message,
                   const FIX::SessionID& session_id) override;

    /// Called before an application message is sent.
    void toApp(FIX::Message& message,
               const FIX::SessionID& session_id) override;

    /// Called when an application message is received.
    /// Delegates to MessageCracker::crack() for type-safe dispatch.
    void fromApp(const FIX::Message& message,
                 const FIX::SessionID& session_id) override;

    // -------------------------------------------------------------------------
    // FIX44::MessageCracker -- typed message handlers
    // -------------------------------------------------------------------------

    /// Handle an incoming NewOrderSingle (MsgType=D) in FIX 4.4.
    /// This is where the exchange logic lives: validate the order,
    /// extract fields, and send back an ExecutionReport.
    void onMessage(const FIX44::NewOrderSingle& message,
                   const FIX::SessionID& session_id) override;

private:
    /// Generate a unique order ID (simple incrementing counter).
    std::string gen_order_id();

    /// Generate a unique execution ID.
    std::string gen_exec_id();

    /// Build and send an ExecutionReport acknowledging a new order.
    void send_execution_report(
        const FIX::Symbol& symbol,
        const FIX::Side& side,
        const FIX::OrderQty& order_qty,
        const FIX::Price& price,
        const FIX::ClOrdID& cl_ord_id,
        const FIX::SessionID& session_id);

    std::atomic<int> order_id_counter_{0};
    std::atomic<int> exec_id_counter_{0};
};
