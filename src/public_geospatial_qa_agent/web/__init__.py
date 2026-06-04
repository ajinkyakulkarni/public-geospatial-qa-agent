"""FastAPI app for the browser UI.

A user types or picks a question, the server streams each stage of
the cycle back as a Server-Sent Event, the page redraws the map after
geocode and after the catalog search. The runner is shared with the
CLI; this package is only the HTTP and SSE plumbing plus a budget
guard.

Intended for local single-user use. There's no auth and the only
abuse guard is a process-wide USD cap. Don't bind to 0.0.0.0 unless
that's truly what you want.
"""
