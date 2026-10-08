from mcp.server.fastmcp import FastMCP


server = FastMCP("test-tools")
DOCUMENTS = {
    "alpha.txt": "普通备忘\n今日晴天\n",
    "beta.txt": "宇宙计划\n先找资料\n再统计行数\n",
}


@server.tool()
def list_documents() -> list[str]:
    """List the available document names."""
    return list(DOCUMENTS)


@server.tool()
def read_document(name: str) -> str:
    """Read the full contents of a document by name."""
    if name not in DOCUMENTS:
        raise ValueError("Unknown document")
    return DOCUMENTS[name]


@server.tool()
def count_document_lines(name: str) -> int:
    """Count the lines in a document by name."""
    if name not in DOCUMENTS:
        raise ValueError("Unknown document")
    return len(DOCUMENTS[name].splitlines())


@server.tool()
def repeat_text(text: str, count: int = 1) -> str:
    """Repeat the supplied text a specified number of times."""
    return text * count


if __name__ == "__main__":
    server.run(transport="stdio")
