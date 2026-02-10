#include "fix_application.h"

#include "quickfix/FileStore.h"
#include "quickfix/FileLog.h"
#include "quickfix/SocketAcceptor.h"
#include "quickfix/SessionSettings.h"

#include <iostream>
#include <stdexcept>

/// Launch the mock exchange (FIX acceptor).
///
/// Reads config/acceptor.cfg, creates a SocketAcceptor with our
/// FixApplication, and listens for incoming FIX connections.
///
/// Usage: fix_acceptor            (uses default config path)
///        fix_acceptor path.cfg   (uses custom config path)
int main(int argc, char* argv[])
{
    try {
        std::string config_path = "config/acceptor.cfg";
        if (argc > 1) {
            config_path = argv[1];
        }

        // SessionSettings reads the .cfg file and parses [DEFAULT] + [SESSION]
        FIX::SessionSettings settings(config_path);

        // FileStoreFactory persists message sequence numbers to disk (in store/)
        // so that sessions can recover after a restart.
        FIX::FileStoreFactory store_factory(settings);

        // FileLogFactory writes FIX message traffic to log files (in log/).
        FIX::FileLogFactory log_factory(settings);

        // Our application that handles business logic.
        FixApplication application;

        // SocketAcceptor listens on the configured port and manages FIX sessions.
        // It runs in a background thread; we block on user input to keep it alive.
        FIX::SocketAcceptor acceptor(application, store_factory, settings, log_factory);

        std::cout << "[Acceptor] Starting mock exchange on port 5001...\n";
        acceptor.start();

        std::cout << "[Acceptor] Press Enter to shut down.\n";
        std::cin.get();

        acceptor.stop();
        std::cout << "[Acceptor] Stopped.\n";

    } catch (const FIX::ConfigError& e) {
        std::cerr << "[Acceptor] Configuration error: " << e.what() << "\n";
        return 1;
    } catch (const std::exception& e) {
        std::cerr << "[Acceptor] Error: " << e.what() << "\n";
        return 1;
    }

    return 0;
}
