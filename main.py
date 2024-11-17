import asyncio
from crewai import Agent, Task, Crew, Process

from textwrap import dedent
from agents import FinancialResearchAgents
from tasks import MarkdownReportCreationTasks


class FinancialCrew:
    def __init__(self, data):
        self.data = data

    async def run(self):
        agents = FinancialResearchAgents()
        tasks = MarkdownReportCreationTasks()

        # AGENTS

        company_research_agent = agents.company_research_agent()
        report_creator = agents.markdown_report_creator()
        chart_creator = agents.chart_creator()
        markdown_writer = agents.markdown_writer()
        stock_analysis_agent = agents.stock_analysis_agent()
        # TASKS
        company_research_task = await tasks.company_research_task(company_research_agent,self.data)
        parse_inputs_task = await tasks.parse_input(report_creator, self.data)
        retrieve_metrics_data_task = await tasks.get_data_from_api(report_creator, [parse_inputs_task])
        create_chart_task = await tasks.create_charts(chart_creator, [retrieve_metrics_data_task])
        create_markdown_file_task = await tasks.write_markdown(markdown_writer, [create_chart_task],self.data)
        stock_analysis_task = await tasks.stock_analysis(stock_analysis_agent,self.data)
        # test_task = tasks.test_task(stock_analysis_agent)



        crew = Crew(
            agents=[
                report_creator,
                company_research_agent,
                
                ],
            tasks=[
                company_research_task,
                
                ],
            verbose=True,
            memory=True,
        )

        result = await crew.kickoff_async()
        return result.raw
    

async def main():
    print("## Welcome to Report Creator Crew")
    print("-------------------------------")
    data = "Marvell Technology"
    mycrew = FinancialCrew(data)
    result = await mycrew.run()
    print("\n\n########################")
    print("## Here is your result:")
    print("########################\n")
    print(result)

if __name__ == "__main__":
    asyncio.run(main())
    