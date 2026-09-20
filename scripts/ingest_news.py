from sqlalchemy import select

from backend.app.config import get_settings
from backend.app.database import SessionLocal
from backend.app.models import NewsArticle, NewsSentiment
from src.data.universe import UNIVERSE
from src.news.provider import NewsDataProvider
from src.nlp.sentiment import VaderFinancialBaseline, associate_symbols


def main():
    key = get_settings().newsdata_api_key
    if not key:
        raise SystemExit("NEWSDATA_API_KEY is required")
    aliases = {x["symbol"]: [x["symbol"], x["name"]] for x in UNIVERSE if x["type"] == "stock"}
    scorer = VaderFinancialBaseline()
    provider = NewsDataProvider(key)
    with SessionLocal() as db:
        for row in provider.latest("India stock market OR NSE OR Sensex"):
            if db.scalar(
                select(NewsArticle.id).where(NewsArticle.external_id == row["external_id"])
            ):
                continue
            row["symbols"] = associate_symbols(
                f"{row['title']} {row.get('description') or ''}", aliases
            )
            article = NewsArticle(**row)
            db.add(article)
            db.flush()
            result = scorer.analyze(f"{article.title}. {article.description or ''}")
            db.add(
                NewsSentiment(
                    article_id=article.id,
                    model=result.model,
                    label=result.label,
                    score=result.score,
                    confidence=result.confidence,
                )
            )
        db.commit()


if __name__ == "__main__":
    main()
