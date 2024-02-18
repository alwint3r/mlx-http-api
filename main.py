import argparse
import uvicorn

def build_arg_parser():
    parser = argparse.ArgumentParser(description='Configure MLX HTTP API server')
    parser.add_argument('--port', '-p', type=int, default=8080, help='Port to listen on')
    parser.add_argument('--host', type=str, default='0.0.0.0', help='Host to listen on')
    parser.add_argument('--dev', '-d', action='store_true', help='Run in development mode')

    return parser

def main():
    parser = build_arg_parser()
    args = parser.parse_args()

    uvicorn.run('app:app', host=args.host, port=args.port, reload=args.dev)

if __name__ == '__main__':
    main()
