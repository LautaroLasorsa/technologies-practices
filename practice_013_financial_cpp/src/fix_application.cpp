#include "fix_application.h"

#include "quickfix/FieldTypes.h"
#include "quickfix/Values.h"

// FIX field includes -- these provide typed field accessors like FIX::Symbol,
// FIX::Side, FIX::OrdType, etc. Each field has a tag number and a type.
#include "quickfix/fix44/ExecutionReport.h"
#include "quickfix/fix44/NewOrderSingle.h"

#include <sstream>
#include <iostream>

// =============================================================================
// Session lifecycle callbacks
// =============================================================================

void FixApplication::onCreate(const FIX::SessionID& session_id)
{
    // TODO(human): Log that a session was created.
    // Print the session_id to stdout so you can see the
    // BeginString, SenderCompID, and TargetCompID.
    //
    // Hint: std::cout << "Session created: " << session_id << std::endl;
}

void FixApplication::onLogon(const FIX::SessionID& session_id)
{
    // TODO(human): Log that a counterparty has logged on.
    // This is called after the Logon handshake completes successfully.
    //
    // Hint: std::cout << "Logon: " << session_id << std::endl;
}

void FixApplication::onLogout(const FIX::SessionID& session_id)
{
    // TODO(human): Log that a counterparty has logged out.
    //
    // Hint: std::cout << "Logout: " << session_id << std::endl;
}

// =============================================================================
// Admin message callbacks (fully implemented -- no user action needed)
// =============================================================================

void FixApplication::toAdmin(FIX::Message& /*message*/,
                             const FIX::SessionID& /*session_id*/)
{
    // No-op: we don't need to modify outgoing admin messages (Logon,
    // Heartbeat, etc.) for this exercise.
}

void FixApplication::fromAdmin(const FIX::Message& /*message*/,
                               const FIX::SessionID& /*session_id*/)
{
    // No-op: we accept all incoming admin messages as-is.
}

void FixApplication::toApp(FIX::Message& /*message*/,
                           const FIX::SessionID& /*session_id*/)
{
    // No-op: we don't modify outgoing application messages.
}

// =============================================================================
// Application message dispatch
// =============================================================================

void FixApplication::fromApp(const FIX::Message& message,
                             const FIX::SessionID& session_id)
{
    // crack() is inherited from FIX44::MessageCracker.
    // It inspects the MsgType field and dispatches to the correct
    // onMessage() overload. If MsgType=D, it calls
    // onMessage(FIX44::NewOrderSingle, session_id).
    crack(message, session_id);
}

// =============================================================================
// NewOrderSingle handler -- the core exercise
// =============================================================================

void FixApplication::onMessage(const FIX44::NewOrderSingle& message,
                               const FIX::SessionID& session_id)
{
    // TODO(human): Extract order fields from the incoming NewOrderSingle.
    //
    // The FIX44::NewOrderSingle message contains typed fields. You extract
    // them by declaring a field variable and calling message.get(field).
    //
    // Extract these five fields:
    //   FIX::Symbol     symbol;       // Tag 55 -- e.g., "AAPL"
    //   FIX::Side       side;         // Tag 54 -- '1'=Buy, '2'=Sell
    //   FIX::OrderQty   order_qty;    // Tag 38 -- e.g., 100
    //   FIX::Price      price;        // Tag 44 -- e.g., 150.25
    //   FIX::ClOrdID    cl_ord_id;    // Tag 11 -- client-assigned order ID
    //
    // Pattern:
    //   FIX::Symbol symbol;
    //   message.get(symbol);
    //
    // After extracting all five, print them to stdout for visibility, then
    // call send_execution_report() with the extracted fields.
    //
    // Reference: https://github.com/quickfix/quickfix/blob/master/examples/executor/C++/Application.cpp

    // --- Your code here ---
}

// =============================================================================
// ExecutionReport builder
// =============================================================================

void FixApplication::send_execution_report(
    const FIX::Symbol& symbol,
    const FIX::Side& side,
    const FIX::OrderQty& order_qty,
    const FIX::Price& price,
    const FIX::ClOrdID& cl_ord_id,
    const FIX::SessionID& session_id)
{
    // TODO(human): Build and send a FIX44::ExecutionReport message.
    //
    // An ExecutionReport (MsgType=8) is the exchange's response to an order.
    // For a new order acknowledgment, you need these fields:
    //
    // Constructor fields (required by FIX44::ExecutionReport):
    //   FIX::OrderID     -- use gen_order_id() to create a unique ID
    //   FIX::ExecID      -- use gen_exec_id() to create a unique ID
    //   FIX::ExecType    -- FIX::ExecType_NEW (character '0') for a new order ack
    //   FIX::OrdStatus   -- FIX::OrdStatus_NEW (character '0') for accepted
    //   FIX::Side        -- echo back the side from the original order
    //   FIX::LeavesQty   -- remaining quantity (same as order_qty for a new order)
    //   FIX::CumQty      -- cumulative filled quantity (0 for a new order)
    //
    // Additional fields to set after construction:
    //   report.set(symbol);
    //   report.set(cl_ord_id);
    //   report.set(order_qty);
    //   report.set(FIX::AvgPx(0));  // average fill price (0 for unfilled)
    //   report.set(price);           // echo the limit price
    //
    // Send:
    //   FIX::Session::sendToTarget(report, session_id);
    //
    // Example construction:
    //   FIX44::ExecutionReport report(
    //       FIX::OrderID(gen_order_id()),
    //       FIX::ExecID(gen_exec_id()),
    //       FIX::ExecType(FIX::ExecType_NEW),
    //       FIX::OrdStatus(FIX::OrdStatus_NEW),
    //       side,
    //       FIX::LeavesQty(order_qty),
    //       FIX::CumQty(0)
    //   );
    //
    // Reference: https://github.com/quickfix/quickfix/blob/master/examples/executor/C++/Application.cpp

    // --- Your code here ---
}

// =============================================================================
// ID generators (fully implemented)
// =============================================================================

std::string FixApplication::gen_order_id()
{
    return std::to_string(++order_id_counter_);
}

std::string FixApplication::gen_exec_id()
{
    return std::to_string(++exec_id_counter_);
}
