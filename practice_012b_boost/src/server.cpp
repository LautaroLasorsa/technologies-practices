#include "server.hpp"
#include "graph_algo.hpp"

#include <boost/asio.hpp>

#include <iostream>
#include <memory>
#include <sstream>
#include <string>

using boost::asio::ip::tcp;

// ============================================================================
// Session -- handles one client connection
// ============================================================================
//
// Key concept: each Session is stored in a shared_ptr. The async callbacks
// capture `shared_from_this()` to keep the Session alive until the async
// operation completes. Without this, the Session would be destroyed when
// the accept handler returns, and the async read/write would access freed
// memory (use-after-free).
//
// This is the standard Asio pattern for managing connection lifetimes.

class Session : public std::enable_shared_from_this<Session> {
public:
  Session(tcp::socket socket, const Graph &graph)
      : socket_(std::move(socket)), graph_(graph) {}

  void start() { do_read(); }

private:
  void do_read() {
    auto self = shared_from_this();

    // ====================================================================
    // TODO(human): Async read a line from the client
    // ====================================================================
    //
    // Use boost::asio::async_read_until to read until '\n':
    //
    //   boost::asio::async_read_until(
    //       socket_,
    //       buffer_,
    //       '\n',
    //       [self](boost::system::error_code ec, std::size_t length) {
    //           if (!ec) {
    //               self->handle_request(length);
    //           } else {
    //               std::cerr << "Read error: " << ec.message() << "\n";
    //           }
    //       }
    //   );
    //
    // Key concepts:
    //   - async_read_until is NON-BLOCKING: it returns immediately
    //   - The lambda is called later when data arrives (or on error)
    //   - Capturing `self` (shared_ptr) keeps the Session alive
    //   - buffer_ is a streambuf that accumulates incoming bytes
    //
    // CP analogy: Think of async operations as "posting a task to a queue"
    // that io_context.run() processes. Similar to how you'd push work
    // onto a BFS queue and process it in a loop.
    //
    // Docs:
    // https://www.boost.org/doc/libs/release/doc/html/boost_asio/reference/async_read_until.html
    // ====================================================================
    boost::asio::async_read_until(
        socket_, buffer_, '\n',
        [self](boost::system::error_code ec, std::size_t length) {
          if (!ec) {
            self->handle_request(length);
          } else {
            std::cerr << "Read error: " << ec.message() << "\n";
          }
        });
    // --- YOUR CODE HERE ---
  }

  void handle_request(std::size_t length) {
    // Extract the line from the streambuf
    std::istream stream(&buffer_);
    std::string line;
    std::getline(stream, line);

    // Parse source vertex
    int source = -1;
    try {
      source = std::stoi(line);
    } catch (...) {
      do_write("Error: invalid vertex ID\n");
      return;
    }

    int n = static_cast<int>(boost::num_vertices(graph_));
    if (source < 0 || source >= n) {
      do_write("Error: vertex out of range [0, " + std::to_string(n) + ")\n");
      return;
    }

    // Run Dijkstra and format response
    auto result = run_dijkstra(graph_, source);
    std::ostringstream oss;
    oss << "Distances from vertex " << source << ":\n";
    for (int i = 0; i < n; ++i) {
      oss << "  " << i << ": " << result.distances[i] << "\n";
    }
    oss << "---END---\n";

    do_write(oss.str());
  }

  void do_write(const std::string &response) {
    auto self = shared_from_this();
    response_ = response;

    // ====================================================================
    // TODO(human): Async write the response to the client
    // ====================================================================
    //
    // Use boost::asio::async_write to send the full response:
    //
    //   boost::asio::async_write(
    //       socket_,
    //       boost::asio::buffer(response_),
    //       [self](boost::system::error_code ec, std::size_t /* bytes */) {
    //           if (ec) {
    //               std::cerr << "Write error: " << ec.message() << "\n";
    //           }
    //           // Session ends here -- socket closes when shared_ptr dies
    //       }
    //   );
    //
    // Note: async_write (not async_write_some) guarantees ALL bytes
    // are sent before the handler is called. async_write_some might
    // only send a partial buffer.
    //
    // Docs:
    // https://www.boost.org/doc/libs/release/doc/html/boost_asio/reference/async_write.html
    // ====================================================================
    boost::asio::async_write(socket_, boost::asio::buffer(response_),
                             [self](boost::system::error_code ec, std::size_t) {
                               if (ec) {
                                 std::cerr << "Write error: " << ec.message()
                                           << "\n";
                               }
                             });
  }

  tcp::socket socket_;
  const Graph &graph_;
  boost::asio::streambuf buffer_;
  std::string response_;
};

// ============================================================================
// Server -- accepts incoming connections
// ============================================================================

class Server {
public:
  Server(boost::asio::io_context &io, uint16_t port, const Graph &graph)
      : acceptor_(io, tcp::endpoint(tcp::v4(), port)), graph_(graph) {
    std::cout << "Server listening on port " << port << "\n";
    do_accept();
  }

private:
  void do_accept() {
    // ====================================================================
    // TODO(human): Async accept a new connection
    // ====================================================================
    //
    // Use acceptor_.async_accept with a lambda:
    //
    //   acceptor_.async_accept(
    //       [this](boost::system::error_code ec, tcp::socket socket) {
    //           if (!ec) {
    //               std::cout << "Client connected: "
    //                         << socket.remote_endpoint() << "\n";
    //               std::make_shared<Session>(std::move(socket),
    //               graph_)->start();
    //           }
    //           // Accept the NEXT connection (loop)
    //           do_accept();
    //       }
    //   );
    //
    // Key concepts:
    //   - async_accept listens for ONE connection, then calls the handler
    //   - To keep accepting, we call do_accept() again inside the handler
    //   - This creates an infinite async accept loop (no recursion on stack)
    //   - Each new session is created via make_shared and starts itself
    //   - The Session's shared_ptr ref count keeps it alive during I/O
    //
    // Pattern: This is called the "async accept loop" -- the fundamental
    // pattern for all Asio-based servers (Beast HTTP, gRPC C++, etc.)
    //
    // Docs:
    // https://www.boost.org/doc/libs/release/doc/html/boost_asio/reference/basic_socket_acceptor/async_accept.html
    // ====================================================================

    acceptor_.async_accept([this](boost::system::error_code ec,
                                  tcp::socket socket) {
      if (!ec) {
        std::cout << "Client connected: " << socket.remote_endpoint() << "\n";
        std::make_shared<Session>(std::move(socket), graph_)->start();
      }
      do_accept();
    });
  }

  tcp::acceptor acceptor_;
  const Graph &graph_;
};

// ============================================================================
// Entry point
// ============================================================================

void run_server(const Graph &graph, uint16_t port, int timeout_seconds) {
  boost::asio::io_context io;

  // Create the server (starts accepting immediately)
  Server server(io, port, graph);

  // ========================================================================
  // TODO(human): Set up a shutdown timer using boost::asio::steady_timer
  // ========================================================================
  //
  // Steps:
  //   1. Create a steady_timer from `io`:
  //        boost::asio::steady_timer timer(io,
  //        std::chrono::seconds(timeout_seconds));
  //
  //   2. Call timer.async_wait with a lambda that stops the io_context:
  //        timer.async_wait([&io](boost::system::error_code ec) {
  //            if (!ec) {
  //                std::cout << "Server timeout -- shutting down\n";
  //                io.stop();
  //            }
  //        });
  //
  // This ensures the server doesn't run forever during practice.
  // io.stop() cancels all pending async operations and makes run() return.
  //
  // Docs:
  // https://www.boost.org/doc/libs/release/doc/html/boost_asio/tutorial/tuttimer2.html
  // ========================================================================

  boost::asio::steady_timer timer(io, std::chrono::seconds(timeout_seconds));
  timer.async_wait([&io](boost::system::error_code ec) {
    if (!ec) {
      std::cout << "Server timeout -- shutting down\n";
      io.stop();
    }
  });

  std::cout << "Server will shut down in " << timeout_seconds << " seconds.\n"
            << "Test with: echo '0' | nc localhost " << port << "\n";

  // io.run() blocks until all async work is done (or io.stop() is called)
  io.run();

  std::cout << "Server stopped.\n";
}
